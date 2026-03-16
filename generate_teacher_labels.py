"""
Fast Knowledge Distillation: Pre-generate teacher predictions (pseudo-labels)
Then train student on cached labels - 5-8x faster than online distillation
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# Configuration  
TEACHER_MODEL_PATH = "./teachers/hausa_teacher_20260311_011457/final_model"
TRAIN_FILE = "./preprocessed_data/hausa/train.csv"
VAL_FILE = "./preprocessed_data/hausa/val.csv"
TEST_FILE = "./preprocessed_data/hausa/test.csv"
USE_PREFIX = True  # mT5 teachers were trained with prefix

OUTPUT_DIR = "./preprocessed_data/hausa_finetuned_teacher_labels"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256
BATCH_SIZE = 16  # Larger batch for inference

def generate_teacher_predictions():
    """Generate teacher predictions for all training data."""
    
    print("\n" + "="*80)
    print("GENERATING TEACHER PSEUDO-LABELS")
    print("="*80)
    print(f"\nTeacher model: {TEACHER_MODEL_PATH}")
    print(f"This will be done ONCE, then student trains much faster")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load teacher model and tokenizer
    print("\nLoading teacher model...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_PATH)
    teacher_model = AutoModelForSeq2SeqLM.from_pretrained(TEACHER_MODEL_PATH)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_model.to(device)
    teacher_model.eval()
    
    print(f"Device: {device}")
    print(f"Teacher parameters: {sum(p.numel() for p in teacher_model.parameters())/1e6:.1f}M")
    
    # Process each split
    for split_name, file_path in [("train", TRAIN_FILE), ("val", VAL_FILE), ("test", TEST_FILE)]:
        print(f"\n{'='*80}")
        print(f"Processing {split_name.upper()} set")
        print(f"{'='*80}")
        
        # Load data
        df = pd.read_csv(file_path)
        print(f"Samples: {len(df)}")
        
        # Generate predictions in batches
        teacher_summaries = []
        
        for i in tqdm(range(0, len(df), BATCH_SIZE), desc=f"Generating {split_name} predictions"):
            batch_texts = df['text'][i:i+BATCH_SIZE].tolist()
            
            # Tokenize (with or without prefix based on how teacher was trained)
            if USE_PREFIX:
                batch_texts = ["summarize: " + text for text in batch_texts]
            
            inputs = tokenizer(
                batch_texts,
                max_length=MAX_INPUT_LENGTH,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            # Generate teacher predictions
            with torch.no_grad():
                outputs = teacher_model.generate(
                    **inputs,
                    max_length=MAX_TARGET_LENGTH,
                    num_beams=6,
                    early_stopping=True
                )
            
            # Decode
            batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            teacher_summaries.extend(batch_summaries)
        
        # Add teacher predictions to dataframe
        df['teacher_summary'] = teacher_summaries
        
        # Save
        output_path = os.path.join(OUTPUT_DIR, f"{split_name}.csv")
        df[['text', 'summary', 'teacher_summary']].to_csv(output_path, index=False)
        
        print(f"Saved to: {output_path}")
        
        # Show sample
        sample = df.iloc[0]
        print(f"\nSample from {split_name}:")
        print(f"Text (first 150 chars): {sample['text'][:150]}...")
        print(f"Ground truth: {sample['summary']}")
        print(f"Teacher pred: {sample['teacher_summary']}")
    
    print("\n" + "="*80)
    print("PSEUDO-LABEL GENERATION COMPLETE!")
    print("="*80)
    print(f"\nAll data saved to: {OUTPUT_DIR}")
    print(f"\nNow the student can train 5-8x faster!")
    print(f"Update train_student.py to use these files:")
    print(f"  TRAIN_FILE = '{OUTPUT_DIR}/train.csv'")
    print(f"  VAL_FILE = '{OUTPUT_DIR}/val.csv'")
    print(f"  TEST_FILE = '{OUTPUT_DIR}/test.csv'")

if __name__ == "__main__":
    generate_teacher_predictions()
