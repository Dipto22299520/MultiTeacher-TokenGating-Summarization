"""
Knowledge Distillation: Train BanglaT5-small student from BanglaT5 teacher
Target: 90%+ retention of teacher performance with temperature=1
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
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback
)
from rouge_score import rouge_scorer

# ============================================================================
# Configuration
# ============================================================================

# Models
TEACHER_MODEL_PATH = "./bangla_t5_teacher_finetuned_20251216_143715/final_model"
STUDENT_MODEL_NAME = "csebuetnlp/banglat5_small"  # Small version
OUTPUT_DIR = "./bangla_t5_student_distilled"

# Data
TRAIN_FILE = "train.csv"
VAL_FILE = "val.csv"
TEST_FILE = "test.csv"

# Tokenization
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256

# Training hyperparameters
BATCH_SIZE = 16  # Larger batch for smaller model
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 5e-5  # Higher LR for student
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
NUM_EPOCHS = 5  # 5 epochs like working code
SEED = 42

# Distillation hyperparameters
TEMPERATURE = 0.8  # Lower temperature = sharper soft targets (working code uses 0.8)
ALPHA = 0.01  # CRITICAL: Very low weight because KL divergence >> CE loss in magnitude
# Working code uses 0.01 - KL divergence naturally has much larger scale than cross entropy

# Evaluation
EVAL_STEPS = 500
SAVE_STEPS = 500

# ============================================================================
# Custom Tokenizer for Bangla ROUGE
# ============================================================================

class SpaceTokenizer:
    """Custom tokenizer that splits by spaces for Bangla text."""
    def tokenize(self, text):
        return text.split()

# ============================================================================
# Knowledge Distillation Trainer
# ============================================================================

class DistillationTrainer(Seq2SeqTrainer):
    """Custom trainer for knowledge distillation with soft label matching."""
    
    def __init__(self, teacher_model=None, temperature=1.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
        if self.teacher is not None:
            self.teacher.eval()
            # Move teacher to same device as student
            self.teacher.to(self.args.device)
            # Freeze teacher
            for param in self.teacher.parameters():
                param.requires_grad = False
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute combined loss: distillation loss + task loss
        Matches working code's simpler approach
        """
        # Get student outputs and loss (hard loss)
        outputs = model(**inputs)
        student_loss = outputs.loss
        
        # If no teacher, just return student loss
        if self.teacher is None:
            return (student_loss, outputs) if return_outputs else student_loss
        
        # Get teacher outputs (soft targets)
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
        
        # Distillation loss (KL divergence on logits)
        student_logits = outputs.logits
        teacher_logits = teacher_outputs.logits
        
        # Apply temperature scaling
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence loss with batchmean reduction (simpler, matches working code)
        distillation_loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combined loss: alpha * soft_loss + (1-alpha) * hard_loss
        # With alpha=0.01, this is ~99% hard loss + ~1% distillation
        loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        return (loss, outputs) if return_outputs else loss

# ============================================================================
# Data preprocessing
# ============================================================================

def preprocess_function(examples, tokenizer):
    """Preprocess function - same as teacher (no prefix)."""
    inputs = examples["text"]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False  # Dynamic padding
    )
    
    labels = tokenizer(
        text_target=examples["summary"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_preds, tokenizer, rouge_scorer_obj):
    """Compute ROUGE metrics for evaluation with Bangla space-based tokenization."""
    preds, labels = eval_preds
    
    # Handle negative values in predictions (replace with pad token)
    preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
    # Replace -100 in labels (used for padding)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute ROUGE scores
    rouge_scores = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }
    
    for pred, label in zip(decoded_preds, decoded_labels):
        # Clean up text
        pred = pred.strip()
        label = label.strip()
        
        if not pred:
            pred = "।"  # Bangla sentence ender as fallback
        if not label:
            label = "।"
        
        scores = rouge_scorer_obj.score(label, pred)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
    
    # Average scores
    result = {
        'rouge1': np.mean(rouge_scores['rouge1']),
        'rouge2': np.mean(rouge_scores['rouge2']),
        'rougeL': np.mean(rouge_scores['rougeL']),
        'rougeLsum': np.mean(rouge_scores['rougeL']),  # Same as rougeL for single sentences
        'gen_len': np.mean([len(pred.split()) for pred in decoded_preds])
    }
    
    return result

# ============================================================================
# Main training function
# ============================================================================

def train_student_model():
    """Train student model using knowledge distillation from teacher."""
    
    print("\n" + "="*80)
    print("KNOWLEDGE DISTILLATION: BANGLAT5 STUDENT FROM TEACHER")
    print("="*80)
    print(f"\nTeacher model: {TEACHER_MODEL_PATH}")
    print(f"Student model: {STUDENT_MODEL_NAME}")
    print(f"Temperature: {TEMPERATURE} (sharper soft targets)")
    print(f"Alpha (distillation weight): {ALPHA} (low because KL >> CE in magnitude)")
    print(f"Max input length: {MAX_INPUT_LENGTH}")
    print(f"Max target length: {MAX_TARGET_LENGTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Using bf16 precision")
    print(f"Target: 90%+ retention (Student ROUGE-L ≥ {0.4058 * 0.9:.4f})")
    
    # Load datasets
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80 + "\n")
    
    train_dataset = load_dataset("csv", data_files={"train": TRAIN_FILE})["train"]
    val_dataset = load_dataset("csv", data_files={"validation": VAL_FILE})["validation"]
    test_dataset = load_dataset("csv", data_files={"test": TEST_FILE})["test"]
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Load tokenizer (use student's tokenizer)
    print("\n" + "="*80)
    print("LOADING TOKENIZER AND MODELS")
    print("="*80 + "\n")
    
    print(f"Loading tokenizer from {STUDENT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME)
    
    # Preprocess datasets
    print("Preprocessing datasets...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train"
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation"
    )
    test_dataset = test_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=test_dataset.column_names,
        desc="Tokenizing test"
    )
    
    print(f"Preprocessed train samples: {len(train_dataset)}")
    print(f"Preprocessed validation samples: {len(val_dataset)}")
    print(f"Preprocessed test samples: {len(test_dataset)}")
    
    # Load teacher model
    print(f"\nLoading TEACHER model from {TEACHER_MODEL_PATH}...")
    teacher_model = AutoModelForSeq2SeqLM.from_pretrained(TEACHER_MODEL_PATH)
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"Teacher parameters: {teacher_params:,} ({teacher_params/1e6:.1f}M)")
    
    # Load student model
    print(f"\nLoading STUDENT model from {STUDENT_MODEL_NAME}...")
    student_model = AutoModelForSeq2SeqLM.from_pretrained(STUDENT_MODEL_NAME)
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"Student parameters: {student_params:,} ({student_params/1e6:.1f}M)")
    print(f"Compression ratio: {teacher_params/student_params:.2f}x smaller")
    
    # Set generation config for student (no min_length - let it decide)
    student_model.config.max_length = MAX_TARGET_LENGTH
    student_model.config.num_beams = 6
    print(f"Student generation config: max_length={MAX_TARGET_LENGTH}, beams=6 (no min_length)")
    
    # Initialize ROUGE scorer
    print("\nInitializing ROUGE scorer with space-based tokenization for Bangla...")
    rouge_scorer_obj = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=False,
        tokenizer=SpaceTokenizer()
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=student_model,
        padding=True,
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
        
        # Training hyperparameters
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        
        # Precision
        bf16=True,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        logging_steps=20,
        
        # Generation during evaluation
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        
        # Optimization
        gradient_checkpointing=False,
        
        # Saving
        save_total_limit=None,  # Keep all checkpoints like working code
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        
        # Other settings
        seed=SEED,
        report_to="none",
        push_to_hub=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # Initialize distillation trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        temperature=TEMPERATURE,
        alpha=ALPHA,
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer, rouge_scorer_obj),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    # Start training
    print("\n" + "="*80)
    print("STARTING KNOWLEDGE DISTILLATION")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nTarget: 90%+ retention of teacher performance")
    print("="*80 + "\n")
    
    train_result = trainer.train()
    
    # Save final model
    print("\nSaving final student model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Training metrics
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Final train loss: {train_result.training_loss:.4f}")
    print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80 + "\n")
    
    test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
    
    print("\nTest Results:")
    print(f"  Test Loss: {test_results['test_loss']:.4f}")
    print(f"  Test ROUGE-1: {test_results['test_rouge1']:.4f}")
    print(f"  Test ROUGE-2: {test_results['test_rouge2']:.4f}")
    print(f"  Test ROUGE-L: {test_results['test_rougeL']:.4f}")
    print(f"  Avg generation length: {test_results['test_gen_len']:.2f} words")
    
    # Save test results
    import json
    results_file = os.path.join(output_dir, "test_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    print(f"\nTest results saved to: {results_file}")
    
    return trainer, test_results

# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    try:
        trainer, test_metrics = train_student_model()
        print("\n" + "="*80)
        print("STUDENT MODEL TRAINING SUCCESSFUL!")
        print("="*80)
        
    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR DURING TRAINING")
        print("="*80)
        print(f"\n{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)