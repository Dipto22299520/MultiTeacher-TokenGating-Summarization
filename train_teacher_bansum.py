"""
Fine-tuning script for mT5 Teacher Models (base/large/XL-Sum)
Supports: mT5-base (~580M), mT5-large (~1.2B), mT5-XLSum
Target: ROUGE-L score of 0.55-0.6

Dataset: BanSum LTE 1000 tokens

GPU UTILIZATION OPTIMIZATION:
- Batch sizes are configured for 70-80% GPU utilization
- mT5-base: batch_size=4, no gradient checkpointing (for ~16-24GB GPU)
- mT5-large: batch_size=2, gradient checkpointing enabled (for ~24-40GB GPU)
- mT5-XLSum: batch_size=4, no gradient checkpointing (for ~16-24GB GPU)

ADJUSTMENTS FOR YOUR GPU:
If GPU utilization is too low (<70%):
  - Increase BATCH_SIZE (try 6, 8, etc.)
  - Decrease GRADIENT_ACCUMULATION_STEPS to maintain effective batch size of ~1024

If GPU runs out of memory:
  - Decrease BATCH_SIZE (try 2, 1)
  - Enable GRADIENT_CHECKPOINTING = True
  - Increase GRADIENT_ACCUMULATION_STEPS to maintain effective batch size

Monitor GPU usage during training with: nvidia-smi -l 1
"""

import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import torch

# Fix PyTorch memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datetime import datetime
from rouge_score import rouge_scorer

# Custom tokenizer for Bangla ROUGE evaluation
class SpaceTokenizer:
    """Space-based tokenizer for Bangla text."""
    def tokenize(self, text):
        return text.split()

# ============================================================================
# MODEL SELECTION - Choose ONE model by uncommenting it
# ============================================================================

# mT5-base - Multilingual T5 base model
MODEL_NAME = "google/mt5-base"
BATCH_SIZE = 16  # Suitable for mT5-base
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch = 32
GRADIENT_CHECKPOINTING = True  # Enable to save memory
LEARNING_RATE = 3e-5

# ============================================================================
# SHARED TRAINING CONFIGURATION
# ============================================================================
MAX_INPUT_LENGTH = 450  # Extended input for BanSum articles
MAX_TARGET_LENGTH = 128  # Extended output for longer summaries
NUM_EPOCHS = 8  # 8 epochs × ~1750 steps = ~14k total steps (matches target)
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
EVAL_STEPS = 999999  # Disabled during training - only at end
SAVE_STEPS = 2000  # Save checkpoints every 2000 steps
LOGGING_STEPS = 100  # Log every 100 steps for more frequent updates
SEED = 42

# ============================================================================
# GPU UTILIZATION MONITORING
# ============================================================================
# To monitor GPU utilization during training, open another terminal and run:
#   nvidia-smi -l 1  (updates every 1 second)
# 
# Target: 70-80% GPU utilization
# - If utilization is consistently below 70%, increase BATCH_SIZE above
# - If you get OOM errors, decrease BATCH_SIZE or enable GRADIENT_CHECKPOINTING
# ============================================================================

# Paths
OUTPUT_DIR = "./banglat5_bansum_finetuned"
BANSUM_FILE = "bansum_lte_1000_tokens.json"

def load_datasets():
    """Load and split BanSum dataset into train, validation, and test sets."""
    print("\n" + "=" * 80)
    print("LOADING BANSUM DATASET")
    print("=" * 80)
    
    # Load JSON file
    print(f"\nLoading data from {BANSUM_FILE}...")
    print("(Large file - this may take a moment...)")
    
    # Standard json loading with explicit encoding
    with open(BANSUM_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    # Using 'main' as text and 'sum1' as summary
    df = pd.DataFrame([{
        'text': item['main'],
        'summary': item['sum2']  # Using sum2 as the target summary
    } for item in data])
    
    print(f"Total samples loaded: {len(df)}")
    
    # Split into train/val/test (80/10/10 split)
    # Shuffle the data first
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    total_size = len(df)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
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
    """Preprocess the data for T5 model."""
    # No prefix - use raw text like the working code
    inputs = [str(text) for text in examples["text"]]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False  # Explicitly False - let data collator handle ALL padding
    )
    
    # Setup the tokenizer for targets (data collator will handle padding)
    targets = [str(summary) for summary in examples["summary"]]
    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False  # Explicitly False - let data collator handle ALL padding
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_preds, tokenizer, rouge_scorer_obj):
    """Compute ROUGE metrics for evaluation with Bangla space-based tokenization."""
    preds, labels = eval_preds
    
    # Replace -100 in both preds and labels (used for padding)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Strip whitespace for cleaner comparison
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # Compute ROUGE scores using rouge_scorer with space-based tokenization
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(decoded_preds, decoded_labels):
        scores = rouge_scorer_obj.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    # Average scores
    result = {
        "rouge1": np.mean(rouge1_scores),
        "rouge2": np.mean(rouge2_scores),
        "rougeL": np.mean(rougeL_scores),
        "rougeLsum": np.mean(rougeL_scores)  # Use same as rougeL for simplicity
    }
    
    # Add prediction length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    
    return result

def train_model():
    """Main training function."""
    print("\n" + "=" * 80)
    print("mT5 TEACHER MODEL FINE-TUNING - BANSUM DATASET")
    print("=" * 80)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Dataset: BanSum LTE 1000 tokens")
    print(f"Max input length: {MAX_INPUT_LENGTH}")
    print(f"Max target length: {MAX_TARGET_LENGTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Gradient checkpointing: {GRADIENT_CHECKPOINTING}")
    print(f"Using bf16 precision (same as working code)")
    
    # GPU info
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("Note: Batch sizes are optimized for 70-80% GPU utilization")
    else:
        print("\nWarning: CUDA not available, training will be slow!")
    
    # Set random seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Load datasets
    datasets = load_datasets()
    
    # Load tokenizer
    print("\n" + "=" * 80)
    print("LOADING TOKENIZER AND MODEL")
    print("=" * 80)
    print(f"\nLoading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Preprocess datasets
    print("\nPreprocessing datasets...")
    tokenized_datasets = datasets.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=datasets["train"].column_names,
        desc="Tokenizing"
    )
    
    print(f"Preprocessed train samples: {len(tokenized_datasets['train'])}")
    print(f"Preprocessed validation samples: {len(tokenized_datasets['validation'])}")
    print(f"Preprocessed test samples: {len(tokenized_datasets['test'])}")
    
    # Load model
    print(f"\nLoading model from {MODEL_NAME}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # Fix generation config to avoid checkpoint save errors
    # The pretrained model may have beam-based parameters with num_beams=1
    print("\nFixing generation config for checkpoint compatibility...")
    if hasattr(model, 'generation_config'):
        model.generation_config.num_beams = 6  # Match training args
        model.generation_config.early_stopping = True
        model.generation_config.length_penalty = 1.2
        print(f"  num_beams: {model.generation_config.num_beams}")
        print(f"  early_stopping: {model.generation_config.early_stopping}")
        print(f"  length_penalty: {model.generation_config.length_penalty}")
    
    # Print model size (no config modifications - keep it simple like working code)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Initialize ROUGE scorer with space-based tokenizer for Bangla
    print("\nInitializing ROUGE scorer with space-based tokenization for Bangla...")
    rouge_scorer_obj = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], 
        use_stemmer=False, 
        tokenizer=SpaceTokenizer()
    )
    
    # Data collator (will handle dynamic padding - use defaults like working code)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    # Check for existing mt5base_bansum directory to resume
    existing_dirs = [d for d in os.listdir('.') if d.startswith('mt5base_bansum_')]
    if existing_dirs:
        output_dir = existing_dirs[0]  # Use the first/only one
        print(f"\n🔄 Found existing training directory: {output_dir}")
        print(f"   Will resume from latest checkpoint")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short_name = MODEL_NAME.split('/')[-1].replace('_', '-')
        output_dir = f"mt5base_bansum_{timestamp}"
        print(f"\n✨ Creating new output directory: {output_dir}")
    
    # Create output and logs directories to avoid TensorBoard errors
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        
        # Training hyperparameters
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,  # Can use larger batch for eval
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        
        # Use bf16 like the working code
        fp16=False,
        bf16=True,
        
        # Evaluation and logging
        eval_strategy="no",  # Disabled during training - only at end
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        logging_dir=f"{output_dir}/logs",
        
        # Generation config for evaluation (used only at end)
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=4,  # For final evaluation only
        
        # Optimization
        gradient_checkpointing=GRADIENT_CHECKPOINTING,  # Enable for large models to reduce memory
        
        # Saving
        save_total_limit=None,  # Save ALL checkpoints
        load_best_model_at_end=False,  # Disabled since no eval during training
        
        # Other settings
        seed=SEED,
        report_to="none",  # Disable reporting to avoid potential issues
        push_to_hub=False,
        dataloader_num_workers=0,  # Critical: set to 0 like working code
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    print("\n" + "=" * 80)
    print("INITIALIZING TRAINER")
    print("=" * 80)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer, rouge_scorer_obj)
    )
    
    # Train - resume from checkpoint if exists
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"TensorBoard logs: {output_dir}/logs")
    print("\nModel: google/mt5-base")
    print("Dataset: BanSum LTE 1000 tokens")
    print("Pre-trained on: Multilingual corpus (includes Bangla)")
    print("Expected improvement: High quality Bangla summarization\n")
    print("To monitor training, run:")
    print(f"tensorboard --logdir {output_dir}/logs")
    print("\n" + "=" * 80)
    
    # Start or resume training
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
        if checkpoints:
            print(f"\n🔄 RESUMING from latest checkpoint in: {output_dir}")
            train_result = trainer.train(resume_from_checkpoint=True)
        else:
            print(f"\n🆕 Starting fresh mT5-base training")
            train_result = trainer.train()
    else:
        print(f"\n🆕 Starting fresh mT5-base training")
        train_result = trainer.train()
    
    # Save the final model
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    trainer.save_model(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("EVALUATING ON VALIDATION SET")
    print("=" * 80)
    
    val_metrics = trainer.evaluate()
    print("\nValidation Results:")
    for key, value in val_metrics.items():
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
    
    test_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"], metric_key_prefix="test")
    print("\nTest Results:")
    for key, value in test_metrics.items():
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
    
    # Get 5 random samples from test set
    test_samples = tokenized_datasets["test"].shuffle(seed=SEED).select(range(5))
    
    # Generate predictions
    predictions = trainer.predict(test_samples)
    decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    
    # Load original data to get text and reference summaries
    with open(BANSUM_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get the test split indices
    total_size = len(data)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    
    # Recreate the shuffle to get correct test samples
    df_full = pd.DataFrame([{
        'text': item['main'],
        'summary': item['sum1']
    } for item in data])
    df_full = df_full.sample(frac=1, random_state=SEED).reset_index(drop=True)
    test_df_full = df_full[train_size + val_size:]
    test_samples_df = test_df_full.sample(n=5, random_state=SEED)
    
    for i, (idx, row) in enumerate(test_samples_df.iterrows()):
        print(f"\n--- Sample {i+1} ---")
        print(f"Text (first 200 chars): {row['text'][:200]}...")
        print(f"\nReference Summary: {row['summary']}")
        print(f"\nGenerated Summary: {decoded_preds[i]}")
        print("-" * 80)
    
    # Save configuration
    config = {
        "model_name": MODEL_NAME,
        "dataset": "BanSum LTE 1000 tokens",
        "max_input_length": MAX_INPUT_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "warmup_steps": WARMUP_STEPS,
        "weight_decay": WEIGHT_DECAY,
        "train_samples": len(tokenized_datasets["train"]),
        "val_samples": len(tokenized_datasets["validation"]),
        "test_samples": len(tokenized_datasets["test"]),
        "final_val_rougeL": val_metrics.get("eval_rougeL", 0),
        "final_test_rougeL": test_metrics.get("test_rougeL", 0),
        "timestamp": timestamp
    }
    
    with open(f"{output_dir}/training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModel saved to: {output_dir}/final_model")
    print(f"Training config saved to: {output_dir}/training_config.json")
    print(f"\nFinal Validation ROUGE-L: {val_metrics.get('eval_rougeL', 0):.4f}")
    print(f"Final Test ROUGE-L: {test_metrics.get('test_rougeL', 0):.4f}")
    
    target_met = test_metrics.get('test_rougeL', 0) >= 0.55
    print(f"\nTarget ROUGE-L (0.55-0.6): {'✓ MET' if target_met else '✗ NOT MET'}")
    
    return trainer, test_metrics

if __name__ == "__main__":
    trainer, test_metrics = train_model()
