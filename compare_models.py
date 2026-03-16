"""
Model Comparison Evaluator

Compare your old model vs new models to see quality improvements.

Usage:
    python compare_models.py --old_model path/to/old --new_model path/to/new --samples 20
"""

import argparse
import json
import random
import unicodedata
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from rouge_score import rouge_scorer
import numpy as np


class CharTokenizer:
    def tokenize(self, text):
        return list(text)


def normalize_bangla(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace('\u200d', '').replace('\u200c', '')
    text = ' '.join(text.split())
    return text.strip()


def load_test_samples(data_file: str, num_samples: int = 20) -> List[Dict]:
    """Load random test samples from the dataset."""
    print(f"Loading samples from {data_file}...")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array")
    
    # Clean and filter
    valid_samples = []
    for item in data:
        text = normalize_bangla(str(item.get("text", "")))
        summary = normalize_bangla(str(item.get("summary", "")))
        
        if len(text) > 100 and len(summary) > 20:
            valid_samples.append({
                "text": text,
                "gold_summary": summary
            })
    
    # Random sample
    if len(valid_samples) > num_samples:
        valid_samples = random.sample(valid_samples, num_samples)
    
    print(f"Loaded {len(valid_samples)} test samples")
    return valid_samples


def generate_summaries(model_path: str, samples: List[Dict], device: str = "cuda") -> List[str]:
    """Generate summaries using a model."""
    print(f"\nGenerating summaries with model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()
    
    summaries = []
    
    with torch.no_grad():
        for i, sample in enumerate(samples):
            text = "summarize bangla news: " + sample["text"]
            
            # Tokenize (truncate to avoid memory issues)
            inputs = tokenizer(
                text,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_length=256,
                min_length=64,
                num_beams=5,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                early_stopping=True
            )
            
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            summaries.append(normalize_bangla(summary))
            
            if (i + 1) % 5 == 0:
                print(f"  Generated {i + 1}/{len(samples)}")
    
    print(f"  Completed: {len(summaries)} summaries")
    return summaries


def compute_rouge_scores(predictions: List[str], references: List[str]) -> Dict:
    """Compute ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], 
                                      use_stemmer=False, 
                                      tokenizer=CharTokenizer())
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': np.mean(rouge1_scores),
        'rouge2': np.mean(rouge2_scores),
        'rougeL': np.mean(rougeL_scores),
        'rouge1_std': np.std(rouge1_scores),
        'rouge2_std': np.std(rouge2_scores),
        'rougeL_std': np.std(rougeL_scores),
    }


def compute_bertscore(predictions: List[str], references: List[str]) -> Dict:
    """Compute BERTScore."""
    try:
        from bert_score import score as bert_score
        
        print("\nComputing BERTScore (this may take a while)...")
        P, R, F1 = bert_score(
            predictions,
            references,
            lang="bn",
            verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item(),
        }
    except ImportError:
        print("⚠️  BERTScore not available (pip install bert-score)")
        return {'f1': 0.0}


def print_comparison_report(
    samples: List[Dict],
    old_summaries: List[str],
    new_summaries: List[str],
    old_scores: Dict,
    new_scores: Dict,
    old_bert: Dict,
    new_bert: Dict
):
    """Print a detailed comparison report."""
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON REPORT")
    print("=" * 80)
    
    # ROUGE Scores
    print("\n📊 ROUGE SCORES:")
    print("-" * 80)
    print(f"{'Metric':<15} {'Old Model':<20} {'New Model':<20} {'Δ Change':<15}")
    print("-" * 80)
    
    for metric in ['rouge1', 'rouge2', 'rougeL']:
        old = old_scores[metric]
        new = new_scores[metric]
        delta = new - old
        delta_pct = (delta / old * 100) if old > 0 else 0
        
        symbol = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
        
        print(f"{metric:<15} {old:.4f} ± {old_scores[metric + '_std']:.4f}    "
              f"{new:.4f} ± {new_scores[metric + '_std']:.4f}    "
              f"{symbol} {delta:+.4f} ({delta_pct:+.1f}%)")
    
    # BERTScore
    if old_bert['f1'] > 0 and new_bert['f1'] > 0:
        print("\n📊 BERTScore (Semantic Similarity):")
        print("-" * 80)
        
        old_f1 = old_bert['f1']
        new_f1 = new_bert['f1']
        delta = new_f1 - old_f1
        delta_pct = (delta / old_f1 * 100) if old_f1 > 0 else 0
        
        print(f"{'F1':<15} {old_f1:.4f}              {new_f1:.4f}              "
              f"{'📈' if delta > 0 else '📉'} {delta:+.4f} ({delta_pct:+.1f}%)")
    
    # Sample Comparisons
    print("\n" + "=" * 80)
    print("SAMPLE COMPARISONS (First 3)")
    print("=" * 80)
    
    for i in range(min(3, len(samples))):
        print(f"\n{'─' * 80}")
        print(f"SAMPLE {i + 1}")
        print(f"{'─' * 80}")
        
        print("\n📄 ARTICLE (first 300 chars):")
        print(samples[i]['text'][:300] + "...")
        
        print("\n✅ GOLD SUMMARY:")
        print(samples[i]['gold_summary'])
        
        print("\n🔴 OLD MODEL:")
        print(old_summaries[i])
        
        print("\n🟢 NEW MODEL:")
        print(new_summaries[i])
    
    # Overall Assessment
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)
    
    rouge_improved = new_scores['rougeL'] > old_scores['rougeL']
    bert_improved = new_bert['f1'] > old_bert['f1'] if new_bert['f1'] > 0 else None
    
    if rouge_improved and (bert_improved is None or bert_improved):
        print("\n🎉 IMPROVEMENT CONFIRMED!")
        print("✅ ROUGE-L improved")
        if bert_improved:
            print("✅ BERTScore improved (better semantic quality)")
        print("\nThe new model shows better quality across metrics.")
    elif rouge_improved:
        print("\n✅ ROUGE-L Improved")
        print("⚠️  BERTScore slightly decreased (check semantic quality manually)")
    else:
        print("\n⚠️  Mixed Results")
        print("Consider:")
        print("  - Training longer")
        print("  - Adjusting hyperparameters")
        print("  - Checking data quality")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare two summarization models")
    parser.add_argument("--old_model", type=str, required=True,
                       help="Path to old/baseline model")
    parser.add_argument("--new_model", type=str, required=True,
                       help="Path to new model to evaluate")
    parser.add_argument("--data_file", type=str, default="bangla_train_combined.json",
                       help="Path to dataset")
    parser.add_argument("--samples", type=int, default=20,
                       help="Number of test samples")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON TOOL")
    print("=" * 80)
    print(f"\nOld Model: {args.old_model}")
    print(f"New Model: {args.new_model}")
    print(f"Test Samples: {args.samples}")
    print(f"Device: {args.device}")
    
    # Load test samples
    samples = load_test_samples(args.data_file, args.samples)
    gold_summaries = [s['gold_summary'] for s in samples]
    
    # Generate with old model
    print("\n" + "=" * 80)
    print("EVALUATING OLD MODEL")
    print("=" * 80)
    old_summaries = generate_summaries(args.old_model, samples, args.device)
    old_rouge = compute_rouge_scores(old_summaries, gold_summaries)
    old_bert = compute_bertscore(old_summaries, gold_summaries)
    
    # Generate with new model
    print("\n" + "=" * 80)
    print("EVALUATING NEW MODEL")
    print("=" * 80)
    new_summaries = generate_summaries(args.new_model, samples, args.device)
    new_rouge = compute_rouge_scores(new_summaries, gold_summaries)
    new_bert = compute_bertscore(new_summaries, gold_summaries)
    
    # Print comparison
    print_comparison_report(
        samples, old_summaries, new_summaries,
        old_rouge, new_rouge, old_bert, new_bert
    )
    
    # Save detailed results
    output_file = "comparison_results.json"
    results = {
        "old_model": args.old_model,
        "new_model": args.new_model,
        "num_samples": len(samples),
        "old_rouge": old_rouge,
        "new_rouge": new_rouge,
        "old_bertscore": old_bert,
        "new_bertscore": new_bert,
        "samples": [
            {
                "text": s['text'][:200] + "...",
                "gold": s['gold_summary'],
                "old_pred": old_summaries[i],
                "new_pred": new_summaries[i]
            }
            for i, s in enumerate(samples[:10])  # Save first 10
        ]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
