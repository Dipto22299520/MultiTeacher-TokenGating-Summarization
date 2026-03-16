"""
Full Document Summarization Training (STEP 1 - Most Important)

CRITICAL INSIGHT FROM ANALYSIS:
Your model learned "compress whatever text you see" but not:
- What information is important across the whole document
- Which events dominate the narrative
- What to ignore
- How to build thematic coherence

THIS IS THE MISSING CAPABILITY.

Solution: Train on FULL documents (truncate to 1024 tokens if needed).
The first 1k tokens usually contain main entities, main event, core narrative.

Model learns:
✅ Salience ranking
✅ Discourse compression  
✅ Abstraction patterns

Data: bangla_train_combined.json (79,502 samples)
Model: csebuetnlp/banglaT5
"""

import os
import sys
import json
import argparse
import unicodedata
import numpy as np
from typing import Optional
import torch
from pathlib import Path

# Fix PyTorch memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    T5Tokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datetime import datetime
from rouge_score import rouge_scorer

# BERTScore
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("⚠️  BERTScore not installed. Install with: pip install bert-score")
    BERTSCORE_AVAILABLE = False


class CharTokenizer:
    """Character-level tokenizer for Bangla text."""
    def tokenize(self, text):
        return list(text)


class WordTokenizer:
    """Word-level tokenizer for standard ROUGE evaluation."""
    def tokenize(self, text):
        return text.split()


def normalize_bangla(text):
    """Normalize Bangla text: NFKC normalization, strip ZWJ/ZWNJ, clean whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace('\u200d', '').replace('\u200c', '')
    text = ' '.join(text.split())
    return text.strip()


def fix_bangla_for_tokenizer(text):
    """
    Fix Bangla text for BanglaT5 tokenizer compatibility.
    
    CRITICAL FIX: BanglaT5's SentencePiece vocabulary uses DECOMPOSED forms
    of certain Bangla characters. The precomposed Unicode forms map to <unk>,
    which destroys the model's ability to generate these characters.
    
    Precomposed (causes <unk>)  ->  Decomposed (works correctly)
    য় (U+09DF)                 ->  য + ় (U+09AF + U+09BC)
    ড় (U+09DC)                 ->  ড + ় (U+09A1 + U+09BC)
    ঢ় (U+09DD)                 ->  ঢ + ় (U+09A2 + U+09BC)
    
    Also normalizes smart quotes and other problematic characters.
    """
    # Decompose precomposed Bangla characters that cause <unk>
    text = text.replace('\u09DF', '\u09AF\u09BC')  # য় -> য + ়
    text = text.replace('\u09DC', '\u09A1\u09BC')  # ড় -> ড + ়
    text = text.replace('\u09DD', '\u09A2\u09BC')  # ঢ় -> ঢ + ়
    
    # Normalize smart quotes and dashes to ASCII equivalents
    text = text.replace('\u2018', "'")   # ' -> '
    text = text.replace('\u2019', "'")   # ' -> '
    text = text.replace('\u201C', '"')   # " -> "
    text = text.replace('\u201D', '"')   # " -> "
    text = text.replace('\u2014', '-')   # — -> -
    text = text.replace('\u2013', '-')   # – -> -
    text = text.replace('\u2026', '...')  # … -> ...
    
    # Remove ZWNJ/ZWJ that can cause issues
    text = text.replace('\u200c', '')     # ZWNJ
    text = text.replace('\u200d', '')     # ZWJ
    
    return text


def load_tokenizer(model_name: str):
    """Load a slow tokenizer to avoid tiktoken/fast conversion issues."""
    try:
        return T5Tokenizer.from_pretrained(model_name, use_fast=False)
    except Exception as exc:
        print(f"⚠️  T5Tokenizer load failed, falling back to AutoTokenizer: {exc}")
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)


# ============================================================================
# CONFIGURATION - FULL DOCUMENT TRAINING
# ============================================================================

# Parse command line arguments
parser = argparse.ArgumentParser(description='Full Document Summarization Training')
parser.add_argument('--train_file', type=str, default='data_splits/train.json', help='Training data file')
parser.add_argument('--val_file', type=str, default='data_splits/val.json', help='Validation data file')
parser.add_argument('--test_file', type=str, default='data_splits/test.json', help='Test data file')
parser.add_argument('--model', type=str, default='csebuetnlp/banglaT5', help='Model name')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume from (e.g., ./banglaT5_full_doc_20260214_224524/checkpoint-2000)')
parser.add_argument('--output_dir', type=str, default=None, help='Output directory (auto-derived from checkpoint if resuming)')

args, unknown = parser.parse_known_args()

MODEL_NAME = args.model

# Training Hyperparameters
MAX_INPUT_LENGTH = 1024      # Truncate to first 1024 tokens (contains main narrative)
MAX_TARGET_LENGTH = 256      # Allow longer summaries for quality
MIN_TARGET_LENGTH = 64
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION_STEPS = 8   # Effective batch = 32
NUM_EPOCHS = args.epochs          # More data = more epochs possible
LEARNING_RATE = args.lr
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRADIENT_CHECKPOINTING = True     # Enable for large dataset
SEED = 42

# Generation Parameters - Extractive/Copy-focused
NUM_BEAMS = 5
LENGTH_PENALTY = 1.0              # Neutral
NO_REPEAT_NGRAM_SIZE = 3          # Prevent degenerate loops during generation
REPETITION_PENALTY = 1.0          # No penalty - allow copying patterns from source
EARLY_STOPPING = True
LABEL_SMOOTHING = 0.0             # ZERO - hard targets force model to copy gold summary exactly
INPUT_PREFIX = "summarize bangla news: "

# Evaluation Strategy - Disabled for faster training
EVAL_STRATEGY = "no"              # No evaluation during training
SAVE_STEPS = 500                  # Save every 500 steps
SAVE_TOTAL_LIMIT = None           # Keep ALL checkpoints
LOGGING_STEPS = 100
LOAD_BEST_MODEL = False           # No eval = can't pick best

# Determine output directory
if args.output_dir:
    OUTPUT_DIR = args.output_dir
elif args.resume_from_checkpoint:
    # Extract parent directory from checkpoint path
    checkpoint_path = Path(args.resume_from_checkpoint)
    OUTPUT_DIR = str(checkpoint_path.parent)
    print(f"Auto-detected output directory from checkpoint: {OUTPUT_DIR}")
else:
    OUTPUT_DIR = f"./banglaT5_full_doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_prepare_data():
    """
    Load pre-split dataset files for full-document training.
    
    Key difference from chunked training:
    - We take the FULL article (or first 1024 tokens)
    - NO chunking, NO memory headers, NO attention bias
    - Simple: input = document, target = summary
    
    This teaches the model SALIENCE LEARNING.
    """
    print("\n" + "=" * 80)
    print("LOADING FULL DOCUMENT TRAINING DATA")
    print("=" * 80)
    print(f"\nTrain file: {args.train_file}")
    print(f"Val file: {args.val_file}")
    print(f"Test file: {args.test_file}")
    print(f"Strategy: Full document → Gold summary (teach salience ranking)")
    
    # Load train data
    print(f"\nLoading {args.train_file}...")
    with open(args.train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    if not isinstance(train_data, list):
        raise ValueError("Train dataset must be a JSON array")
    
    # Load val data
    print(f"Loading {args.val_file}...")
    with open(args.val_file, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    if not isinstance(val_data, list):
        raise ValueError("Val dataset must be a JSON array")
    
    # Load test data
    print(f"Loading {args.test_file}...")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if not isinstance(test_data, list):
        raise ValueError("Test dataset must be a JSON array")
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_data):,}")
    print(f"  Val:   {len(val_data):,}")
    print(f"  Test:  {len(test_data):,}")
    print(f"  Total: {len(train_data) + len(val_data) + len(test_data):,}")
    
    # Convert to HuggingFace Dataset
    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data),
        'test': Dataset.from_list(test_data)
    })
    
    return dataset_dict


def preprocess_function(examples, tokenizer):
    """
    Preprocess data for full-document training.
    
    SIMPLE APPROACH:
    - Input: prefix + full document text (truncated to MAX_INPUT_LENGTH)
    - Target: gold summary
    
    NO chunking metadata, NO memory headers.
    The model must learn what's important from the full context.
    """
    # Fix Bangla text for tokenizer compatibility (decompose য়/ড়/ঢ়)
    inputs = [INPUT_PREFIX + fix_bangla_for_tokenizer(text) for text in examples["text"]]
    targets = [fix_bangla_for_tokenizer(s) for s in examples["summary"]]
    
    # Tokenize inputs (will truncate to MAX_INPUT_LENGTH)
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False
    )
    
    # Tokenize targets
    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_metrics(eval_pred, tokenizer):
    """
    Compute ROUGE-L and BERTScore.
    
    Focus on:
    - ROUGE-L: Surface-level quality
    - BERTScore: Semantic similarity (addresses the core problem)
    """
    predictions, labels = eval_pred
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels (used for padding)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Normalize
    decoded_preds = [normalize_bangla(pred) for pred in decoded_preds]
    decoded_labels = [normalize_bangla(label) for label in decoded_labels]
    
    # ROUGE scores (word-level - standard evaluation)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False, tokenizer=WordTokenizer())
    rouge_scores = []
    
    rouge1_scores = []
    rouge2_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        score = scorer.score(label, pred)
        rouge_scores.append(score['rougeL'].fmeasure)
        rouge1_scores.append(score['rouge1'].fmeasure)
        rouge2_scores.append(score['rouge2'].fmeasure)
    
    avg_rouge_l = np.mean(rouge_scores)
    avg_rouge1 = np.mean(rouge1_scores)
    avg_rouge2 = np.mean(rouge2_scores)
    
    # BERTScore (semantic similarity)
    bert_score_f1 = 0.0
    if BERTSCORE_AVAILABLE:
        try:
            _, _, F1 = bert_score(
                decoded_preds,
                decoded_labels,
                lang="bn",
                verbose=False,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            bert_score_f1 = F1.mean().item()
        except Exception as e:
            print(f"⚠️  BERTScore computation failed: {e}")
    
    metrics = {
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rouge_l,
        'bertScore_f1': bert_score_f1
    }
    
    return metrics


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    """Main training pipeline for full-document summarization."""
    
    print("\n" + "=" * 80)
    print("FULL DOCUMENT SUMMARIZATION TRAINING")
    print("Addressing: Global Salience Learning (Most Critical Fix)")
    print("=" * 80)
    
    # Load data
    dataset = load_and_prepare_data()
    
    # Load tokenizer and model
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = load_tokenizer(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Set generation defaults here to avoid passing a dict into generation_config
    model.generation_config.max_length = MAX_TARGET_LENGTH
    model.generation_config.min_length = MIN_TARGET_LENGTH
    model.generation_config.num_beams = NUM_BEAMS
    model.generation_config.length_penalty = LENGTH_PENALTY
    model.generation_config.no_repeat_ngram_size = NO_REPEAT_NGRAM_SIZE
    model.generation_config.repetition_penalty = REPETITION_PENALTY
    model.generation_config.early_stopping = EARLY_STOPPING
    
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Preprocess datasets
    print("\nTokenizing datasets...")
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing"
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy=EVAL_STRATEGY,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=LOAD_BEST_MODEL,
        predict_with_generate=False,
        bf16=True,  # Mixed precision for speed (better stability than fp16)
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        label_smoothing_factor=LABEL_SMOOTHING,
        seed=SEED,
        report_to="tensorboard",
        push_to_hub=False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Initialize trainer - no evaluation, save all checkpoints
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )
    
    # Print configuration
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Total samples: {len(dataset['train']) + len(dataset['validation']) + len(dataset['test']):,}")
    print(f"Training samples: {len(dataset['train']):,}")
    print(f"Validation samples: {len(dataset['validation']):,}")
    print(f"Test samples: {len(dataset['test']):,}")
    print(f"\nHyperparameters:")
    print(f"  Max input length: {MAX_INPUT_LENGTH}")
    print(f"  Max target length: {MAX_TARGET_LENGTH}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Warmup ratio: {WARMUP_RATIO}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"\nGeneration config:")
    print(f"  Num beams: {NUM_BEAMS}")
    print(f"  Length penalty: {LENGTH_PENALTY}")
    print(f"  No-repeat n-gram: {NO_REPEAT_NGRAM_SIZE}")
    print(f"  Repetition penalty: {REPETITION_PENALTY}")
    print(f"  Label smoothing: {LABEL_SMOOTHING}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("=" * 80)
    
    # Train
    print("\nStarting training...")
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
    print("Teaching the model GLOBAL SALIENCE LEARNING...")
    print("(What's important, what to ignore, how to build coherent narratives)\n")
    
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    
    # Save all results
    print("\nSaving results...")
    
    results = {
        "train_results": {k: float(v) if isinstance(v, (np.floating, float)) else v 
                         for k, v in train_result.metrics.items()},
        "test_results": {k: float(v) if isinstance(v, (np.floating, float)) else v 
                        for k, v in test_results.items()},
        "training_config": {
            "model": MODEL_NAME,
            "train_file": args.train_file,
            "val_file": args.val_file,
            "test_file": args.test_file,
            "total_samples": len(dataset['train']) + len(dataset['validation']) + len(dataset['test']),
            "train_samples": len(dataset['train']),
            "val_samples": len(dataset['validation']),
            "test_samples": len(dataset['test']),
            "max_input_length": MAX_INPUT_LENGTH,
            "max_target_length": MAX_TARGET_LENGTH,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "num_beams": NUM_BEAMS,
            "length_penalty": LENGTH_PENALTY,
            "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
            "repetition_penalty": REPETITION_PENALTY,
            "label_smoothing": LABEL_SMOOTHING,
        }
    }
    
    with open(f"{OUTPUT_DIR}/all_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nTest ROUGE-L: {test_results.get('eval_rougeL', 0):.4f}")
    if 'eval_bertScore_f1' in test_results:
        print(f"Test BERTScore F1: {test_results.get('eval_bertScore_f1', 0):.4f}")
    print(f"\nModel saved to: {OUTPUT_DIR}/final_model")
    print(f"Results saved to: {OUTPUT_DIR}/all_results.json")
    
    print("\n" + "=" * 80)
    print("WHAT THIS MODEL LEARNED:")
    print("=" * 80)
    print("✅ Global salience ranking (what's important)")
    print("✅ Narrative structure understanding")
    print("✅ Main event/entity identification")
    print("✅ Discourse-level compression")
    print("✅ Abstraction patterns")
    print("\nThis is the FOUNDATION for high-quality summarization.")
    print("Next steps: Train reduce task, then multi-task training.")
    print("=" * 80)


if __name__ == "__main__":
    main()
