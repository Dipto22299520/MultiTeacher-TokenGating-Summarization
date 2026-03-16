"""
Fast Student Training: Uses pre-generated teacher labels
No teacher model loaded during training = 5-8x faster
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
from rouge_score import rouge_scorer

# ============================================================================
# Configuration
# ============================================================================

# Models - NO TEACHER MODEL LOADED (uses cached predictions)
STUDENT_MODEL_NAME = "google/mt5-small"
OUTPUT_DIR = "./students/hausa_student_fast"

# Data - with teacher labels pre-generated
TRAIN_FILE = "././preprocessed_data/hausa_finetuned_teacher_labels/train.csv"
VAL_FILE = "././preprocessed_data/hausa_finetuned_teacher_labels/val.csv"
TEST_FILE = "././preprocessed_data/hausa_finetuned_teacher_labels/test.csv"
USE_PREFIX = True  # Match teacher prefix usage

# Tokenization
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256

# Training hyperparameters
BATCH_SIZE = 8  # Smaller batch = more update steps per epoch on tiny datasets
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch = 16
LEARNING_RATE = 3e-4  # Higher LR — student needs to learn fast on little data
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 50  # Was 500 — amharic only has ~55 steps/epoch, must warm up fast
NUM_EPOCHS = 10  # Was 3 — small datasets need more passes; early stopping will stop if done
SEED = 42

# Mixed supervision: optimize mostly toward gold summaries while still pulling
# the student toward the teacher's style and task-specific behavior.
GOLD_LOSS_WEIGHT = 0.7
TEACHER_LOSS_WEIGHT = 0.3

# Evaluation
EVAL_STEPS = 200  # Was 2000 — many languages never reached a single eval!
SAVE_STEPS = 200

# ============================================================================
# Custom Tokenizer for ROUGE
# ============================================================================

class SpaceTokenizer:
    """Custom tokenizer that splits by spaces."""
    def tokenize(self, text):
        return text.split()

# ============================================================================
# Data preprocessing
# ============================================================================

def build_inputs(examples, tokenizer):
    """Tokenize source articles using the same prefix as the teacher."""
    # Add prefix only if teacher was trained with it
    if USE_PREFIX:
        inputs = ["summarize: " + str(text) for text in examples["text"]]
    else:
        inputs = [str(text) for text in examples["text"]]
    return tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False
    )


def preprocess_train_function(examples, tokenizer):
    """Train on both gold summaries and teacher summaries."""
    model_inputs = build_inputs(examples, tokenizer)

    gold_labels = tokenizer(
        text_target=[str(s) for s in examples["summary"]],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False
    )
    teacher_labels = tokenizer(
        text_target=[str(s) for s in examples["teacher_summary"]],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False
    )

    model_inputs["labels"] = gold_labels["input_ids"]
    model_inputs["teacher_labels"] = teacher_labels["input_ids"]
    return model_inputs


def preprocess_eval_function(examples, tokenizer):
    """Evaluate against gold summaries only."""
    model_inputs = build_inputs(examples, tokenizer)

    gold_labels = tokenizer(
        text_target=[str(s) for s in examples["summary"]],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False
    )

    model_inputs["labels"] = gold_labels["input_ids"]
    return model_inputs


class MixedDataCollator:
    """Pad input features plus optional teacher label targets."""

    def __init__(self, tokenizer, model=None, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.model = model
        self.label_pad_token_id = label_pad_token_id

    def _pad_label_field(self, sequences):
        padded = self.tokenizer.pad(
            [{"input_ids": seq} for seq in sequences],
            padding=True,
            return_tensors="pt"
        )["input_ids"]
        padded[padded == self.tokenizer.pad_token_id] = self.label_pad_token_id
        return padded

    def __call__(self, features):
        labels = [feature.pop("labels") for feature in features]
        teacher_labels = None
        if "teacher_labels" in features[0]:
            teacher_labels = [feature.pop("teacher_labels") for feature in features]

        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = self._pad_label_field(labels)

        if teacher_labels is not None:
            batch["teacher_labels"] = self._pad_label_field(teacher_labels)

        return batch


class MixedDistillationTrainer(Seq2SeqTrainer):
    """Train student on a weighted mix of gold and teacher targets."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        teacher_labels = inputs.pop("teacher_labels", None)

        gold_outputs = model(**inputs)
        loss = gold_outputs.loss

        if teacher_labels is not None:
            teacher_inputs = {
                key: value for key, value in inputs.items() if key != "labels"
            }
            teacher_inputs["labels"] = teacher_labels
            teacher_outputs = model(**teacher_inputs)
            loss = GOLD_LOSS_WEIGHT * loss + TEACHER_LOSS_WEIGHT * teacher_outputs.loss

        if return_outputs:
            return loss, gold_outputs
        return loss

def compute_metrics(eval_preds, tokenizer, rouge_scorer_obj):
    """Compute ROUGE metrics."""
    preds, labels = eval_preds
    
    # Handle padding
    preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute ROUGE scores
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for pred, label in zip(decoded_preds, decoded_labels):
        pred = pred.strip() or "।"
        label = label.strip() or "।"
        
        scores = rouge_scorer_obj.score(label, pred)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
    
    result = {
        'rouge1': np.mean(rouge_scores['rouge1']),
        'rouge2': np.mean(rouge_scores['rouge2']),
        'rougeL': np.mean(rouge_scores['rougeL']),
        'rougeLsum': np.mean(rouge_scores['rougeL']),
        'gen_len': np.mean([len(pred.split()) for pred in decoded_preds])
    }
    
    return result

# ============================================================================
# Main training function
# ============================================================================

def train_student_fast():
    """Train student using pre-generated teacher labels."""
    
    print("\n" + "="*80)
    print("FAST STUDENT TRAINING (Pre-generated Teacher Labels)")
    print("="*80)
    print(f"\nStudent model: {STUDENT_MODEL_NAME}")
    print(f"Training on cached teacher predictions (NO teacher model loaded)")
    print(f"Expected speedup: 5-8x faster than online distillation")
    print(f"Max input length: {MAX_INPUT_LENGTH}")
    print(f"Max target length: {MAX_TARGET_LENGTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    
    # Load datasets
    print("\n" + "="*80)
    print("LOADING DATASETS WITH TEACHER LABELS")
    print("="*80 + "\n")
    
    train_dataset = load_dataset("csv", data_files={"train": TRAIN_FILE})["train"]
    val_dataset = load_dataset("csv", data_files={"validation": VAL_FILE})["validation"]
    test_dataset = load_dataset("csv", data_files={"test": TEST_FILE})["test"]
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Check required columns
    required_train_columns = {"summary", "teacher_summary"}
    missing_train_columns = required_train_columns - set(train_dataset.column_names)
    if missing_train_columns:
        print("\n" + "="*80)
        print("ERROR: required training columns not found!")
        print("="*80)
        print(f"\nMissing: {sorted(missing_train_columns)}")
        print("Please ensure gold summaries and teacher predictions are present.")
        print("If teacher_summary is missing, run: python generate_teacher_labels.py")
        sys.exit(1)

    if "summary" not in val_dataset.column_names or "summary" not in test_dataset.column_names:
        print("\n" + "="*80)
        print("ERROR: summary column not found in validation/test data!")
        print("="*80)
        print("Validation and test must keep gold summaries for proper model selection.")
        sys.exit(1)
    
    # Load tokenizer
    print("\n" + "="*80)
    print("LOADING TOKENIZER AND MODEL")
    print("="*80 + "\n")
    
    print(f"Loading tokenizer from {STUDENT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME)
    
    # Preprocess datasets
    print("Preprocessing datasets...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_train_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train"
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_eval_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing val"
    )
    test_dataset = test_dataset.map(
        lambda x: preprocess_eval_function(x, tokenizer),
        batched=True,
        remove_columns=test_dataset.column_names,
        desc="Tokenizing test"
    )
    
    print(f"Preprocessed train: {len(train_dataset)}")
    print(f"Preprocessed val: {len(val_dataset)}")
    print(f"Preprocessed test: {len(test_dataset)}")
    
    # Load student model
    print(f"\nLoading STUDENT model from {STUDENT_MODEL_NAME}...")
    student_model = AutoModelForSeq2SeqLM.from_pretrained(STUDENT_MODEL_NAME)
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"Student parameters: {student_params:,} ({student_params/1e6:.1f}M)")
    
    # Set generation config properly (not on model.config)
    from transformers import GenerationConfig
    generation_config = GenerationConfig.from_pretrained(STUDENT_MODEL_NAME)
    generation_config.max_length = MAX_TARGET_LENGTH
    generation_config.num_beams = 6
    student_model.generation_config = generation_config
    
    # Initialize ROUGE scorer
    print("\nInitializing ROUGE scorer...")
    rouge_scorer_obj = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=False,
        tokenizer=SpaceTokenizer()
    )
    
    # Data collator
    data_collator = MixedDataCollator(
        tokenizer=tokenizer,
        model=student_model,
        label_pad_token_id=-100
    )
    
    # Training arguments
    print("\n" + "="*80)
    print("INITIALIZING TRAINER")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{OUTPUT_DIR}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        bf16=True,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        logging_steps=20,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        gradient_checkpointing=False,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        seed=SEED,
        report_to="none",
        push_to_hub=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = MixedDistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer, rouge_scorer_obj),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    # Start training
    print("\n" + "="*80)
    print("STARTING FAST TRAINING")
    print("="*80)
    print(f"\nOutput: {output_dir}")
    print(f"Expected speed: ~2-3s/iteration (vs 14-16s with online distillation)")
    print("="*80 + "\n")
    
    train_result = trainer.train()
    
    # Save model
    print("\nSaving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate on test
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80 + "\n")
    
    test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
    
    print("\nTest Results:")
    print(f"  ROUGE-1: {test_results['test_rouge1']:.4f}")
    print(f"  ROUGE-2: {test_results['test_rouge2']:.4f}")
    print(f"  ROUGE-L: {test_results['test_rougeL']:.4f}")
    
    # Save results
    import json
    with open(f"{output_dir}/test_results.json", 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"Model saved to: {output_dir}")
    
    return trainer, test_results

if __name__ == "__main__":
    try:
        trainer, test_metrics = train_student_fast()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
