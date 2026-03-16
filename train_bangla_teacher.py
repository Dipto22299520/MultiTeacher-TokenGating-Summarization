"""
Production-Grade Fine-tuning script for BanglaT5 Model
 
OBJECTIVES:
1. Higher ROUGE-L scores
2. Better BERTScore
3. Excellent semantic scores
4. Better at generation quality
5. Error-free production grade model

Model: csebuetnlp/banglaT5 (Optimized for Bangla summarization)
Data: Bangla articles ≤1000 tokens for efficient training
"""

import os
import json
import unicodedata
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
import torch
from sklearn.model_selection import train_test_split

# Fix PyTorch memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from datasets import Dataset, DatasetDict
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

# BERTScore for semantic evaluation
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("⚠️  BERTScore not installed. Install with: pip install bert-score")
    BERTSCORE_AVAILABLE = False

# Character-level tokenizer for Bangla ROUGE evaluation
# Space-based tokenization is WRONG for Bangla (postpositions, clitics,
# compound words, punctuation sticking to words break LCS subsequences).
# Character-level ROUGE is standard for Bangla/Chinese/morphologically-rich languages.
class CharTokenizer:
    """Character-level tokenizer for Bangla text."""
    def tokenize(self, text):
        return list(text)


def normalize_bangla(text):
    """Normalize Bangla text: NFKC normalization, strip ZWJ/ZWNJ, clean whitespace."""
    text = unicodedata.normalize("NFKC", text)
    # Remove zero-width characters that break scoring
    text = text.replace('\u200d', '').replace('\u200c', '')
    text = ' '.join(text.split())  # Normalize whitespace
    return text.strip()

# ============================================================================
# PRODUCTION-GRADE CONFIGURATION
# ============================================================================

# Model Configuration
MODEL_NAME = "csebuetnlp/banglaT5"

# Data Configuration
DATA_FILE = "bangla_train_lte_1000.json"
TRAIN_SPLIT = 0.85
VAL_SPLIT = 0.10
TEST_SPLIT = 0.05

# Training Hyperparameters (Optimized for Quality)
MAX_INPUT_LENGTH = 1024  # Full coverage for articles ≤1000 tokens
MAX_TARGET_LENGTH = 192  # Bangla summaries need more tokens due to morphology
BATCH_SIZE = 4  # Optimal for banglaT5
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch = 32 (stable training)
NUM_EPOCHS = 15  # More epochs for better convergence
LEARNING_RATE = 5e-5  # Conservative for production quality
WARMUP_RATIO = 0.1  # 10% warmup
WEIGHT_DECAY = 0.01
GRADIENT_CHECKPOINTING = False  # Disabled for speed
SEED = 42

# Generation Parameters (Balanced for quality + semantics)
# High beams + strong no_repeat encourages copying, hurts paraphrasing & BERTScore
NUM_BEAMS = 4  # Balanced beam search
LENGTH_PENALTY = 1.0  # Neutral length preference
NO_REPEAT_NGRAM_SIZE = 2  # Mild repetition control (3 is too aggressive)
REPETITION_PENALTY = 1.2  # Penalize redundant clauses
EARLY_STOPPING = True
LABEL_SMOOTHING = 0.1  # Prevent overconfident decoding

# Evaluation Strategy
EVAL_STRATEGY = "steps"
EVAL_STEPS = 500  # Evaluate every 500 steps
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = None  # Save all checkpoints
LOGGING_STEPS = 50
LOAD_BEST_MODEL = True
METRIC_FOR_BEST_MODEL = "eval_loss"  # Use loss during training (fast, no generation)

# Output Directory
OUTPUT_DIR = f"./banglaT5_production_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_split_data():
    """Load JSON data and split into train/val/test sets."""
    print("\n" + "=" * 80)
    print("LOADING AND SPLITTING DATA")
    print("=" * 80)
    
    # Load JSON file
    print(f"\nLoading data from: {DATA_FILE}")
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Validate data
    valid_data = []
    for item in data:
        if 'text' in item and 'summary' in item:
            if item['text'] and item['summary']:  # Not empty
                valid_data.append(item)
    
    print(f"Valid samples after cleaning: {len(valid_data)}")
    
    # Convert to DataFrame for splitting
    df = pd.DataFrame(valid_data)
    
    # First split: train + val vs test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=TEST_SPLIT, 
        random_state=SEED,
        shuffle=True
    )
    
    # Second split: train vs val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT),
        random_state=SEED,
        shuffle=True
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df[['text', 'summary']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'summary']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'summary']])
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    return dataset_dict

def preprocess_function(examples, tokenizer):
    """Preprocess data for T5 model."""
    # BanglaT5 is T5-family trained with English task prefixes.
    # English prefix converges faster and grounds semantics better.
    prefix = "summarize: "
    inputs = [prefix + str(text) for text in examples["text"]]
    
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False  # Dynamic padding by data collator
    )
    
    # Tokenize targets
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
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(eval_preds, tokenizer, rouge_scorer_obj):
    """
    Compute ROUGE metrics only during training (fast evaluation).
    BERTScore and semantic similarity will be checked later after training.
    - ROUGE-1, ROUGE-2, ROUGE-L (character-level tokenization)
    - Generation length
    """
    preds, labels = eval_preds
    
    # Handle padding (-100 → pad_token_id so decoder can process them)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels (skip_special_tokens removes pad tokens)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Normalize Bangla text before scoring (NFKC + ZWJ removal)
    decoded_preds = [normalize_bangla(pred) for pred in decoded_preds]
    decoded_labels = [normalize_bangla(label) for label in decoded_labels]
    
    # Compute ROUGE scores with character-level tokenization
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
    
    # Generation length statistics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    
    return result

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model():
    """Main training function with production-grade setup."""
    print("\n" + "=" * 80)
    print("PRODUCTION-GRADE BANGLAT5 FINE-TUNING")
    print("=" * 80)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Data: {DATA_FILE}")
    print(f"\nTraining Configuration:")
    print(f"  Max input length: {MAX_INPUT_LENGTH}")
    print(f"  Max target length: {MAX_TARGET_LENGTH}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Warmup ratio: {WARMUP_RATIO}")
    print(f"\nGeneration Configuration:")
    print(f"  Num beams: {NUM_BEAMS}")
    print(f"  Length penalty: {LENGTH_PENALTY}")
    print(f"  No repeat ngram: {NO_REPEAT_NGRAM_SIZE}")
    print(f"  Repetition penalty: {REPETITION_PENALTY}")
    print(f"  Label smoothing: {LABEL_SMOOTHING}")
    
    # GPU info
    if torch.cuda.is_available():
        print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠️  WARNING: CUDA not available, training will be slow!")
    
    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Load and split data
    datasets = load_and_split_data()
    
    # Load tokenizer
    print("\n" + "=" * 80)
    print("LOADING TOKENIZER AND MODEL")
    print("=" * 80)
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    
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
    
    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # Configure generation settings
    print("\nConfiguring generation parameters...")
    if hasattr(model, 'generation_config'):
        model.generation_config.num_beams = NUM_BEAMS
        model.generation_config.length_penalty = LENGTH_PENALTY
        model.generation_config.no_repeat_ngram_size = NO_REPEAT_NGRAM_SIZE
        model.generation_config.repetition_penalty = REPETITION_PENALTY
        model.generation_config.early_stopping = EARLY_STOPPING
        model.generation_config.max_length = MAX_TARGET_LENGTH
        print(f"  ✓ Generation config updated")
    
    # Model info
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    
    # Initialize ROUGE scorer with character-level tokenization for Bangla
    print("\n✓ Initializing ROUGE scorer with character-level tokenization for Bangla")
    rouge_scorer_obj = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=False,
        tokenizer=CharTokenizer()
    )
    
    # Data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/logs", exist_ok=True)
    print(f"\n✓ Output directory: {OUTPUT_DIR}")
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        
        # Training
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        
        # Precision — check bf16 support at runtime to avoid silent fallback
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        
        # Label smoothing prevents overconfident decoding
        label_smoothing_factor=LABEL_SMOOTHING,
        
        # Evaluation and saving
        eval_strategy=EVAL_STRATEGY,
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=LOAD_BEST_MODEL,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        greater_is_better=False,  # Lower loss is better
        
        # Logging
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        
        # Disable generation during training for speed (only compute loss)
        # Will enable generation only for final evaluation after training
        predict_with_generate=False,
        
        # Optimization
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        
        # Other
        seed=SEED,
        report_to="tensorboard",
        push_to_hub=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # Initialize trainer with early stopping
    print("\n" + "=" * 80)
    print("INITIALIZING TRAINER")
    print("=" * 80)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        # No compute_metrics during training - just use loss for speed
        # Will compute ROUGE manually after training with generation enabled
        compute_metrics=None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]  # Stop if no improvement for 5 evals
    )
    
    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"\n✓ Training will save best model based on {METRIC_FOR_BEST_MODEL} (lower is better)")
    print(f"✓ Evaluation uses loss only (no generation) for speed - ROUGE computed after training")
    print(f"✓ Early stopping enabled (patience=5)")
    print(f"\nTo monitor training:")
    print(f"  tensorboard --logdir {OUTPUT_DIR}/logs")
    print("\n" + "=" * 80)
    
    try:
        train_result = trainer.train()
        
        # Save final model
        print("\n" + "=" * 80)
        print("SAVING FINAL MODEL")
        print("=" * 80)
        
        final_model_path = f"{OUTPUT_DIR}/final_model"
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"✓ Model saved to: {final_model_path}")
        
        # Save training metrics
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
    
    # Enable generation for evaluation
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
    
    # Generate sample predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)
    
    # Get 5 random samples
    test_samples = tokenized_datasets["test"].shuffle(seed=SEED).select(range(min(5, len(tokenized_datasets["test"]))))
    
    # Get original data for display
    original_test = datasets["test"].shuffle(seed=SEED).select(range(min(5, len(datasets["test"]))))
    
    # Generate predictions
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
        "data_file": DATA_FILE,
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
        print(f"\n\n❌ Training failed with error: {e}")
        raise
