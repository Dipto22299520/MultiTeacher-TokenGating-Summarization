"""
Quick evaluation script for a specific checkpoint.
Usage: python eval_checkpoint.py --checkpoint ./banglaT5_full_doc_20260214_224524/checkpoint-4000
"""

import os
import sys
import json
import argparse
import unicodedata
import numpy as np
import torch
from pathlib import Path

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from datasets import Dataset
from transformers import (
    T5Tokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from rouge_score import rouge_scorer

# BERTScore
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("⚠️  BERTScore not installed.")
    BERTSCORE_AVAILABLE = False


class CharTokenizer:
    def tokenize(self, text):
        return list(text)


def normalize_bangla(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace('\u200d', '').replace('\u200c', '')
    text = ' '.join(text.split())
    return text.strip()


def load_tokenizer(model_name: str):
    try:
        return T5Tokenizer.from_pretrained(model_name, use_fast=False)
    except Exception as exc:
        print(f"⚠️  T5Tokenizer load failed, falling back to AutoTokenizer: {exc}")
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [normalize_bangla(pred) for pred in decoded_preds]
    decoded_labels = [normalize_bangla(label) for label in decoded_labels]
    
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False, tokenizer=CharTokenizer())
    rouge_scores = []
    
    for pred, label in zip(decoded_preds, decoded_labels):
        score = scorer.score(label, pred)
        rouge_scores.append(score['rougeL'].fmeasure)
    
    avg_rouge_l = np.mean(rouge_scores)
    
    # BERTScore
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
    
    return {
        'rougeL': avg_rouge_l,
        'bertScore_f1': bert_score_f1
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate a checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--test_file', type=str, default='data_splits/test.json', help='Test data file')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("CHECKPOINT EVALUATION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test file: {args.test_file}")
    
    # Load test data
    print(f"\nLoading test data...")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"Test samples: {len(test_data):,}")
    
    # Load model and tokenizer from checkpoint
    print(f"\nLoading model from checkpoint...")
    tokenizer = load_tokenizer(args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
    
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Prepare test dataset
    MAX_INPUT_LENGTH = 1024
    MAX_TARGET_LENGTH = 256
    INPUT_PREFIX = "summarize bangla news: "
    
    def preprocess_function(examples):
        inputs = [INPUT_PREFIX + text for text in examples["text"]]
        targets = examples["summary"]
        
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
    
    print("\nTokenizing test data...")
    test_dataset = Dataset.from_list(test_data)
    tokenized_test = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=test_dataset.column_names,
        desc="Tokenizing"
    )
    
    # Create trainer for evaluation
    training_args = Seq2SeqTrainingArguments(
        output_dir="./eval_temp",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=5,
        bf16=True,
        report_to="none"
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer)
    )
    
    # Evaluate
    print("\nEvaluating...")
    results = trainer.evaluate(tokenized_test)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"ROUGE-L: {results['eval_rougeL']:.4f} ({results['eval_rougeL']*100:.2f}%)")
    if 'eval_bertScore_f1' in results:
        print(f"BERTScore F1: {results['eval_bertScore_f1']:.4f} ({results['eval_bertScore_f1']*100:.2f}%)")
    print(f"Loss: {results['eval_loss']:.4f}")
    print("=" * 80)
    
    # Save results
    checkpoint_name = Path(args.checkpoint).name
    output_file = f"eval_results_{checkpoint_name}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
