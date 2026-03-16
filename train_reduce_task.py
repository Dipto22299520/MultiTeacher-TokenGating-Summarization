"""
Reduce Task Training (STEP 2 — Paper-Quality)

THE PROBLEM:
  Your reduce step: chunk_summaries → final_summary
  But the model was NEVER trained on this mapping!
  During inference, it behaves unpredictably.

THE SOLUTION (Research-Grade):
  1. Use Step 1 teacher model to generate REAL chunk summaries
     (generate_reduce_data.py does this)
  2. Train on: teacher_chunk_summaries → gold_summary
  3. Corruption augmentation for robustness (shuffle, drop, duplicate)

  This matches the train-time input distribution to inference-time inputs.
  Combined with augmentation, the model learns:
    - Merging multiple summaries into one
    - Deduplication of repeated info across chunks
    - Order-invariance (shuffled chunks)
    - Missing-info tolerance (dropped chunks)
    - Contradiction resolution

  Very few papers do this — strong research contribution.

Prerequisites:
  python generate_reduce_data.py   # generates reduce_data/reduce_{train,val,test}.json

Usage:
  python train_reduce_task.py
  python train_reduce_task.py --data_dir reduce_data --epochs 10
  python train_reduce_task.py --resume_from banglaT5_reduce_task_xxx/checkpoint-500
"""

import os
import json
import argparse
import unicodedata
import numpy as np
import torch
from datetime import datetime

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from datasets import Dataset, DatasetDict
from transformers import (
    T5Tokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from rouge_score import rouge_scorer


class WordTokenizer:
    """Word-level tokenizer for standard ROUGE evaluation."""
    def tokenize(self, text):
        return text.split()


def fix_bangla_for_tokenizer(text):
    """Decompose precomposed Bangla chars that cause <unk> in BanglaT5."""
    text = text.replace('\u09DF', '\u09AF\u09BC')  # য় -> য + ়
    text = text.replace('\u09DC', '\u09A1\u09BC')  # ড় -> ড + ়
    text = text.replace('\u09DD', '\u09A2\u09BC')  # ঢ় -> ঢ + ়
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201C', '"').replace('\u201D', '"')
    text = text.replace('\u2014', '-').replace('\u2013', '-')
    text = text.replace('\u2026', '...')
    text = text.replace('\u200c', '').replace('\u200d', '')
    return text


def normalize_bangla(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace('\u200d', '').replace('\u200c', '')
    text = ' '.join(text.split())
    return text.strip()


def load_tokenizer(model_name: str):
    """Load a slow tokenizer to avoid tiktoken/fast conversion issues."""
    try:
        return T5Tokenizer.from_pretrained(model_name, use_fast=False)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)


# ============================================================================
# CONFIGURATION
# ============================================================================

parser = argparse.ArgumentParser(description='Reduce Task Training (Paper-Quality)')
parser.add_argument('--data_dir', type=str, default='reduce_data',
                    help='Directory with reduce_{train,val,test}.json from generate_reduce_data.py')
parser.add_argument('--model', type=str,
                    default='./banglaT5_full_doc_20260215_123349/checkpoint-7000',
                    help='Base model (Step 1 checkpoint)')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--resume_from', type=str, default=None,
                    help='Resume training from a checkpoint directory')

args, _ = parser.parse_known_args()

MODEL_NAME = args.model
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 256
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = args.epochs
LEARNING_RATE = args.lr
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRADIENT_CHECKPOINTING = True
SEED = 42
LABEL_SMOOTHING = 0.0
INPUT_PREFIX = "summarize multiple summaries: "

EVAL_STRATEGY = "no"
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = None
LOGGING_STEPS = 100

OUTPUT_DIR = f"./banglaT5_reduce_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ============================================================================
# DATA LOADING — Pre-generated teacher chunk summaries
# ============================================================================

def load_reduce_data(data_dir: str) -> DatasetDict:
    """
    Load pre-generated reduce data from generate_reduce_data.py.

    Each sample has:
      - text: concatenated teacher-generated chunk summaries with [CHUNK] markers
      - summary: gold summary
      - augmentation: clean | shuffle | drop | duplicate
      - num_chunks: number of chunks
    """
    print("\n" + "=" * 80)
    print("LOADING TEACHER-GENERATED REDUCE DATA")
    print("=" * 80)
    print(f"Data directory: {data_dir}")

    splits = {}
    for split_name in ['train', 'val', 'test']:
        path = os.path.join(data_dir, f"reduce_{split_name}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run generate_reduce_data.py first:\n"
                f"  python generate_reduce_data.py --output_dir {data_dir}"
            )

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Show augmentation breakdown
        aug_counts = {}
        for item in data:
            aug = item.get("augmentation", "clean")
            aug_counts[aug] = aug_counts.get(aug, 0) + 1

        hf_split_name = 'validation' if split_name == 'val' else split_name
        splits[hf_split_name] = Dataset.from_list(data)

        print(f"\n  {split_name}: {len(data):,} samples")
        print(f"    Augmentation: {aug_counts}")

    total = sum(len(s) for s in splits.values())
    print(f"\n  Total: {total:,} samples")

    return DatasetDict(splits)


def preprocess_function(examples, tokenizer):
    """Tokenize reduce task data with Bangla fix applied."""
    inputs = [INPUT_PREFIX + fix_bangla_for_tokenizer(str(text)) for text in examples["text"]]
    targets = [fix_bangla_for_tokenizer(str(s)) for s in examples["summary"]]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False
    )

    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ============================================================================
# EVALUATION
# ============================================================================

def compute_metrics(eval_pred, tokenizer):
    """Compute word-level ROUGE-L."""
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [normalize_bangla(pred) for pred in decoded_preds]
    decoded_labels = [normalize_bangla(label) for label in decoded_labels]

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False, tokenizer=WordTokenizer())
    rouge_scores = []

    for pred, label in zip(decoded_preds, decoded_labels):
        score = scorer.score(label, pred)
        rouge_scores.append(score['rougeL'].fmeasure)

    return {'rougeL': np.mean(rouge_scores)}


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("REDUCE TASK TRAINING (Paper-Quality)")
    print("Teaching: teacher_chunk_summaries → gold_summary")
    print("With robustness augmentation (shuffle, drop, duplicate)")
    print("=" * 80)

    # Load pre-generated reduce data
    dataset = load_reduce_data(args.data_dir)

    # Load model from Step 1 checkpoint
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = load_tokenizer(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    print(f"Model parameters: {model.num_parameters():,}")

    # Tokenize
    print("\nTokenizing datasets...")
    columns_to_remove = [c for c in dataset["train"].column_names
                         if c in ("text", "summary", "augmentation", "num_chunks")]
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=columns_to_remove,
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
        load_best_model_at_end=False,
        predict_with_generate=False,
        bf16=True,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        label_smoothing_factor=LABEL_SMOOTHING,
        seed=SEED,
        report_to="tensorboard",
        push_to_hub=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )

    # Print config
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Task: Reduce (teacher_chunk_summaries → gold_summary)")
    print(f"Base model: {MODEL_NAME}")
    print(f"Training samples: {len(dataset['train']):,}")
    print(f"  (includes augmented: shuffle, drop, duplicate)")
    print(f"Validation samples: {len(dataset['validation']):,}")
    print(f"Test samples: {len(dataset['test']):,}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Label smoothing: {LABEL_SMOOTHING}")
    print(f"Input prefix: '{INPUT_PREFIX}'")
    print("=" * 80)

    # Train
    print("\nStarting training...")
    print("Teaching the model to merge real chunk summaries into coherent final summaries...\n")

    resume_ckpt = args.resume_from
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

    # Save results
    results = {
        "train_results": {k: float(v) if isinstance(v, (np.floating, float)) else v
                         for k, v in train_result.metrics.items()},
        "training_config": {
            "task": "reduce_teacher_chunks",
            "base_model": MODEL_NAME,
            "data_dir": args.data_dir,
            "train_samples": len(dataset['train']),
            "val_samples": len(dataset['validation']),
            "test_samples": len(dataset['test']),
            "epochs": NUM_EPOCHS,
            "lr": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRADIENT_ACCUMULATION_STEPS,
            "label_smoothing": LABEL_SMOOTHING,
            "input_prefix": INPUT_PREFIX,
            "augmentation": "shuffle + drop + duplicate",
        }
    }

    with open(f"{OUTPUT_DIR}/all_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Model saved to: {OUTPUT_DIR}/final_model")
    print(f"Results saved to: {OUTPUT_DIR}/all_results.json")
    print("\nYour reduce phase is now TRAINED on real teacher-generated data!")
    print("Integrate into pipeline: use this model for the reduce step.")
    print("=" * 80)


if __name__ == "__main__":
    main()
