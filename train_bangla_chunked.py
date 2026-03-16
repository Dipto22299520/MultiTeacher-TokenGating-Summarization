"""
Production-Grade Fine-tuning for BanglaT5 on Chunked Long Articles

Based on train_bangla_teacher.py (which achieved ROUGE-L ~0.465 on ≤1000 data).

Key additions for >1000 token articles:
  - Solution 1: Sentence-aligned chunking (via prepare_gt1000_training_data.py)
  - Solution 2: Sliding overlap between chunks
  - Solution 3: Memory-aware chunk headers injected in training data
  - Solution 4: Chunk-aware attention bias (optional, toggle with USE_ATTENTION_BIAS)
  - Combined training on both ≤1000 and chunked >1000 data
  - MapReduce inference pipeline (see inference_pipeline.py)

Model: csebuetnlp/banglaT5
Data: bangla_train_combined.json (lte_1000 + chunked gt_1000)
"""

import os
import json
import unicodedata
import numpy as np
from typing import Optional
import torch

# Fix PyTorch memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datetime import datetime
from rouge_score import rouge_scorer

# Solution 4 imports
from attention_bias import ChunkAwareT5, add_chunk_boundary_token, CHUNK_BOUNDARY_TOKEN

# BERTScore
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("⚠️  BERTScore not installed. Install with: pip install bert-score")
    BERTSCORE_AVAILABLE = False


# Character-level tokenizer for Bangla ROUGE evaluation
class CharTokenizer:
    """Character-level tokenizer for Bangla text."""
    def tokenize(self, text):
        return list(text)


def normalize_bangla(text):
    """Normalize Bangla text: NFKC normalization, strip ZWJ/ZWNJ, clean whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace('\u200d', '').replace('\u200c', '')
    text = ' '.join(text.split())
    return text.strip()


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model
MODEL_NAME = "csebuetnlp/banglaT5"

# Data — use pre-split data to prevent leakage (no article in multiple splits)
DATA_DIR = "data_splits"
TRAIN_FILE = "data_splits/train.jsonl"
VAL_FILE = "data_splits/val.jsonl"
TEST_FILE = "data_splits/test.jsonl"

# ===========================================
# Solution 4 Toggle
# ===========================================
# Set to True to enable chunk-aware attention bias (Solution 4)
# Set to False to use vanilla T5 with Solutions 1-3 only
USE_ATTENTION_BIAS = False
ATTENTION_BIAS_ALPHA = 0.5       # Strength of cross-chunk attention dampening
ATTENTION_BIAS_LEARNABLE = True  # Whether alpha is a learnable parameter

# Training Hyperparameters (same proven config as lte_1000 model)
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 256
MIN_TARGET_LENGTH = 64
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8   # Effective batch = 32
NUM_EPOCHS = 25
LEARNING_RATE = 3e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRADIENT_CHECKPOINTING = False
SEED = 42

# Generation Parameters
NUM_BEAMS = 6
LENGTH_PENALTY = 1.2
NO_REPEAT_NGRAM_SIZE = 3
REPETITION_PENALTY = 1.15
EARLY_STOPPING = True
LABEL_SMOOTHING = 0.05
INPUT_PREFIX = "summarize bangla news: "

# Evaluation Strategy
EVAL_STRATEGY = "steps"
EVAL_STEPS = 500
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 3  # Keep only 3 best checkpoints
LOGGING_STEPS = 50
LOAD_BEST_MODEL = True
METRIC_FOR_BEST_MODEL = "eval_rougeL"
GREATER_IS_BETTER = True

# Output Directory
bias_tag = "_attnbias" if USE_ATTENTION_BIAS else ""
OUTPUT_DIR = f"./banglaT5_chunked{bias_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ============================================================================
# DATA LOADING
# ============================================================================

def load_presplit_data():
    """Load pre-split data (no leakage - articles not shared across splits)."""
    print("\n" + "=" * 80)
    print("LOADING PRE-SPLIT DATA (NO LEAKAGE)")
    print("=" * 80)
    
    print(f"\nLoading from: {DATA_DIR}/")
    print(f"  Train: {TRAIN_FILE}")
    print(f"  Val:   {VAL_FILE}")
    print(f"  Test:  {TEST_FILE}")
    
    # Load each split separately
    train_dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
    val_dataset = load_dataset("json", data_files=VAL_FILE, split="train")
    test_dataset = load_dataset("json", data_files=TEST_FILE, split="train")

    def clean_split(split_dataset, split_name):
        """Filter noisy pairs and remove duplicates inside each split."""
        keep_indices = []
        seen = set()

        for idx, item in enumerate(split_dataset):
            text = normalize_bangla(str(item.get("text", "")))
            summary = normalize_bangla(str(item.get("summary", "")))

            if len(text) < 80 or len(summary) < 16:
                continue

            text_words = text.split()
            summary_words = summary.split()
            if len(text_words) < 20 or len(summary_words) < 5:
                continue

            if len(summary_words) > 140:
                continue

            pair_key = (text, summary)
            if pair_key in seen:
                continue

            seen.add(pair_key)
            keep_indices.append(idx)

        cleaned = split_dataset.select(keep_indices)
        removed = len(split_dataset) - len(cleaned)
        print(f"  {split_name}: removed {removed} noisy/duplicate rows")
        return cleaned

    print("\nCleaning splits (filter + dedup)...")
    train_dataset = clean_split(train_dataset, "Train")
    val_dataset = clean_split(val_dataset, "Val")
    test_dataset = clean_split(test_dataset, "Test")
    
    print(f"\nLoaded samples:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")
    print(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    
    return DatasetDict({'train': train_dataset, 'validation': val_dataset, 'test': test_dataset})


def preprocess_function(examples, tokenizer):
    """Preprocess data for T5 model."""
    chunk_indices = examples.get("chunk_index", [None] * len(examples["text"]))
    total_chunks = examples.get("total_chunks", [None] * len(examples["text"]))
    has_memory_flags = examples.get("has_memory", [None] * len(examples["text"]))

    inputs = []
    for text, chunk_idx, total_chunk, has_memory in zip(
        examples["text"], chunk_indices, total_chunks, has_memory_flags
    ):
        metadata_parts = []
        if chunk_idx is not None and total_chunk is not None:
            metadata_parts.append(f"chunk {int(chunk_idx) + 1}/{int(total_chunk)}")
        if has_memory is True:
            metadata_parts.append("with prior context")

        metadata_prefix = f" [{' | '.join(metadata_parts)}]" if metadata_parts else ""
        inputs.append(f"{INPUT_PREFIX}{metadata_prefix} {str(text)}")
    
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False
    )
    
    targets = [str(summary) for summary in examples["summary"]]
    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(eval_preds, tokenizer, rouge_scorer_obj):
    """Compute ROUGE metrics with character-level tokenization for Bangla."""
    preds, labels = eval_preds
    
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [normalize_bangla(pred) for pred in decoded_preds]
    decoded_labels = [normalize_bangla(label) for label in decoded_labels]
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(decoded_preds, decoded_labels):
        scores = rouge_scorer_obj.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    result = {
        "rouge1": np.mean(rouge1_scores),
        "rouge2": np.mean(rouge2_scores),
        "rougeL": np.mean(rougeL_scores),
        "rougeLsum": np.mean(rougeL_scores)
    }
    
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)

    pred_word_lens = [max(1, len(pred.split())) for pred in decoded_preds]
    ref_word_lens = [max(1, len(ref.split())) for ref in decoded_labels]
    length_ratios = [p / r for p, r in zip(pred_word_lens, ref_word_lens)]
    result["len_ratio"] = float(np.mean(length_ratios))

    repetition_rates = []
    for pred in decoded_preds:
        words = pred.split()
        if len(words) < 2:
            repetition_rates.append(0.0)
            continue
        unique_words = len(set(words))
        repetition_rates.append(1.0 - (unique_words / len(words)))
    result["rep_rate"] = float(np.mean(repetition_rates))
    
    return result


# ============================================================================
# CUSTOM TRAINER FOR CHUNK-AWARE MODEL
# ============================================================================

class ChunkAwareTrainer(Seq2SeqTrainer):
    """
    Custom trainer that handles the ChunkAwareT5 model wrapper.
    
    The main difference: when saving/loading, we need to handle the
    attention bias parameters separately from the base T5 model.
    """
    
    def __init__(self, chunk_model=None, **kwargs):
        self._chunk_model = chunk_model
        super().__init__(**kwargs)
    
    def save_model(self, output_dir=None, _internal_call=False):
        """Save both the base model and attention bias parameters."""
        if output_dir is None:
            output_dir = self.args.output_dir
        
        if self._chunk_model is not None:
            os.makedirs(output_dir, exist_ok=True)
            self._chunk_model.save_pretrained(output_dir)
        else:
            super().save_model(output_dir, _internal_call)


# ============================================================================
# TRAINING
# ============================================================================

def train_model():
    """Main training function."""
    print("\n" + "=" * 80)
    print("CHUNKED BANGLAT5 FINE-TUNING")
    print("Solutions: 1 (Sentence Chunking) + 2 (Overlap) + 3 (Memory Headers)"
          + (" + 4 (Attention Bias)" if USE_ATTENTION_BIAS else ""))
    print("=" * 80)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Attention Bias: {'ENABLED (α={:.2f})'.format(ATTENTION_BIAS_ALPHA) if USE_ATTENTION_BIAS else 'DISABLED'}")
    print(f"\nTraining Configuration:")
    print(f"  Max input length: {MAX_INPUT_LENGTH}")
    print(f"  Max target length: {MAX_TARGET_LENGTH}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"\nGeneration Configuration:")
    print(f"  Num beams: {NUM_BEAMS}")
    print(f"  Length penalty: {LENGTH_PENALTY}")
    print(f"  Min target length: {MIN_TARGET_LENGTH}")
    print(f"  No repeat ngram: {NO_REPEAT_NGRAM_SIZE}")
    print(f"  Repetition penalty: {REPETITION_PENALTY}")
    print(f"  Label smoothing: {LABEL_SMOOTHING}")
    
    # GPU info
    if torch.cuda.is_available():
        print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠️  WARNING: CUDA not available!")
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Load pre-split data
    datasets = load_presplit_data()
    
    # Load tokenizer
    print("\n" + "=" * 80)
    print("LOADING TOKENIZER AND MODEL")
    print("=" * 80)
    
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    
    # Load model (with or without attention bias)
    chunk_model = None
    if USE_ATTENTION_BIAS:
        print(f"\n✓ Loading ChunkAwareT5 with attention bias (α={ATTENTION_BIAS_ALPHA})")
        chunk_model = ChunkAwareT5.from_pretrained(
            MODEL_NAME,
            tokenizer,
            alpha=ATTENTION_BIAS_ALPHA,
            learnable_alpha=ATTENTION_BIAS_LEARNABLE
        )
        model = chunk_model.base_model  # For Trainer compatibility
        
        # The tokenizer now has the [CHUNK_BOUNDARY] token
        print(f"  Tokenizer vocabulary size: {len(tokenizer)}")
        print(f"  [CHUNK_BOUNDARY] token ID: {tokenizer.convert_tokens_to_ids(CHUNK_BOUNDARY_TOKEN)}")
    else:
        print(f"\nLoading model: {MODEL_NAME}")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # Preprocess datasets
    print("\nPreprocessing datasets...")
    tokenized_datasets = datasets.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=datasets["train"].column_names,
        desc="Tokenizing"
    )
    
    print(f"\nTokenized samples:")
    print(f"  Train: {len(tokenized_datasets['train'])}")
    print(f"  Validation: {len(tokenized_datasets['validation'])}")
    print(f"  Test: {len(tokenized_datasets['test'])}")
    
    # Configure generation
    if hasattr(model, 'generation_config'):
        model.generation_config.num_beams = NUM_BEAMS
        model.generation_config.length_penalty = LENGTH_PENALTY
        model.generation_config.no_repeat_ngram_size = NO_REPEAT_NGRAM_SIZE
        model.generation_config.repetition_penalty = REPETITION_PENALTY
        model.generation_config.early_stopping = EARLY_STOPPING
        model.generation_config.min_length = MIN_TARGET_LENGTH
        model.generation_config.max_length = MAX_TARGET_LENGTH
    
    # Model info
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    if chunk_model:
        bias_params = sum(p.numel() for p in chunk_model.attention_bias.parameters())
        print(f"  Attention bias parameters: {bias_params}")
    
    # ROUGE scorer
    rouge_scorer_obj = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=False,
        tokenizer=CharTokenizer()
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/logs", exist_ok=True)
    print(f"\n✓ Output directory: {OUTPUT_DIR}")
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        
        label_smoothing_factor=LABEL_SMOOTHING,
        
        eval_strategy=EVAL_STRATEGY,
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=LOAD_BEST_MODEL,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        greater_is_better=GREATER_IS_BETTER,
        
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=NUM_BEAMS,
        
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        
        seed=SEED,
        report_to="tensorboard",
        push_to_hub=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    print("\n" + "=" * 80)
    print("INITIALIZING TRAINER")
    print("=" * 80)
    
    if USE_ATTENTION_BIAS and chunk_model:
        trainer = ChunkAwareTrainer(
            chunk_model=chunk_model,
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer, rouge_scorer_obj),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer, rouge_scorer_obj),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )
    
    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"\n✓ Best model based on {METRIC_FOR_BEST_MODEL} ({'higher' if GREATER_IS_BETTER else 'lower'} is better)")
    print(f"✓ Generation-aware evaluation enabled during training")
    print(f"✓ Early stopping (patience=5)")
    print(f"\nMonitor: tensorboard --logdir {OUTPUT_DIR}/logs")
    print("=" * 80)
    
    try:
        train_result = trainer.train()
        
        # Save final model
        print("\n" + "=" * 80)
        print("SAVING FINAL MODEL")
        print("=" * 80)
        
        final_model_path = f"{OUTPUT_DIR}/final_model"
        
        if USE_ATTENTION_BIAS and chunk_model:
            chunk_model.save_pretrained(final_model_path)
        else:
            trainer.save_model(final_model_path)
        
        tokenizer.save_pretrained(final_model_path)
        print(f"✓ Model saved to: {final_model_path}")
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        raise
    
    # Evaluate on validation set WITH GENERATION
    print("\n" + "=" * 80)
    print("EVALUATING ON VALIDATION SET (with generation)")
    print("=" * 80)
    
    trainer.args.predict_with_generate = True
    trainer.args.generation_max_length = MAX_TARGET_LENGTH
    trainer.args.generation_num_beams = NUM_BEAMS
    trainer.compute_metrics = lambda eval_preds: compute_metrics(eval_preds, tokenizer, rouge_scorer_obj)
    
    val_metrics = trainer.evaluate()
    print("\n✓ Validation Results:")
    for key, value in sorted(val_metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    trainer.log_metrics("eval", val_metrics)
    trainer.save_metrics("eval", val_metrics)
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80)
    
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_datasets["test"],
        metric_key_prefix="test"
    )
    print("\n✓ Test Results:")
    for key, value in sorted(test_metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)
    
    # Sample predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)
    
    test_samples = tokenized_datasets["test"].shuffle(seed=SEED).select(
        range(min(5, len(tokenized_datasets["test"])))
    )
    original_test = datasets["test"].shuffle(seed=SEED).select(
        range(min(5, len(datasets["test"])))
    )
    
    predictions = trainer.predict(test_samples)
    decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    
    for i in range(len(decoded_preds)):
        print(f"\n{'='*40} Sample {i+1} {'='*40}")
        print(f"\n📄 Original Text (first 200 chars):")
        print(f"   {original_test[i]['text'][:200]}...")
        print(f"\n🎯 Reference Summary:")
        print(f"   {original_test[i]['summary']}")
        print(f"\n🤖 Generated Summary:")
        print(f"   {decoded_preds[i]}")
    
    # Save configuration
    config = {
        "model_name": MODEL_NAME,
        "data_dir": DATA_DIR,
        "solutions_enabled": {
            "solution_1_sentence_chunking": True,
            "solution_2_sliding_overlap": True,
            "solution_3_memory_headers": True,
            "solution_4_attention_bias": USE_ATTENTION_BIAS,
        },
        "attention_bias_alpha": ATTENTION_BIAS_ALPHA if USE_ATTENTION_BIAS else None,
        "max_input_length": MAX_INPUT_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "num_beams": NUM_BEAMS,
        "length_penalty": LENGTH_PENALTY,
        "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
        "train_samples": len(tokenized_datasets["train"]),
        "val_samples": len(tokenized_datasets["validation"]),
        "test_samples": len(tokenized_datasets["test"]),
        "final_metrics": {
            "val_rougeL": val_metrics.get("eval_rougeL", 0),
            "test_rougeL": test_metrics.get("test_rougeL", 0)
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    config_path = f"{OUTPUT_DIR}/training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Model saved: {final_model_path}")
    print(f"📁 Config saved: {config_path}")
    print(f"\n📊 Final Metrics:")
    print(f"  Validation ROUGE-L: {val_metrics.get('eval_rougeL', 0):.4f}")
    print(f"  Test ROUGE-L: {test_metrics.get('test_rougeL', 0):.4f}")
    
    return trainer, test_metrics


if __name__ == "__main__":
    try:
        trainer, test_metrics = train_model()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        raise
