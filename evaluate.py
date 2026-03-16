"""
Evaluation Script
==================
Evaluates trained student models on the test set.
Computes: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore, and teacher disagreement analysis.

Usage:
    python evaluate.py --model-dir LMI/student_outputs/ewad_full_20260219_120000/best_model
    python evaluate.py --model-dir LMI/student_outputs/ewad_full_20260219_120000/best_model --analysis
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


class SpaceTokenizer:
    """Space-based tokenizer for Bangla ROUGE evaluation."""
    def tokenize(self, text):
        return text.split()


def load_test_data(quick_eval=False):
    """Load test split of BanSum. If quick_eval=True, return only QUICK_EVAL_SAMPLES."""
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    np.random.seed(SEED)
    indices = np.random.permutation(len(all_data))
    all_data = [all_data[i] for i in indices]
    
    total = len(all_data)
    train_end = int(TRAIN_SPLIT * total)
    val_end = train_end + int(VAL_SPLIT * total)
    
    test_data = all_data[val_end:]
    
    if MAX_SAMPLES is not None:
        test_data = test_data[:MAX_SAMPLES]
    
    if quick_eval:
        test_data = test_data[:QUICK_EVAL_SAMPLES]
        print(f"Quick eval: {len(test_data)} samples (of {QUICK_EVAL_SAMPLES} requested)")
    else:
        print(f"Full eval: {len(test_data)} test samples")
    
    return test_data


def generate_summaries(model, tokenizer, test_data, batch_size=EVAL_BATCH_SIZE, max_new_tokens=256):
    """Generate summaries for test data."""
    model.eval()
    predictions = []
    
    for i in tqdm(range(0, len(test_data), batch_size), desc="Generating"):
        batch_data = test_data[i:i+batch_size]
        texts = [item[DATASET_TEXT_KEY] for item in batch_data]
        
        # Tokenize
        inputs = tokenizer(
            texts,
            truncation=True,
            max_length=STUDENT_MAX_INPUT_TOKENS,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=EVAL_NUM_BEAMS,
                    length_penalty=EVAL_LENGTH_PENALTY,
                    early_stopping=True,
                    do_sample=False,
                )
        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM at sample {i}. Retrying with greedy decoding...")
            torch.cuda.empty_cache()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                )
        
        # Decode — strip input tokens
        for j, output in enumerate(outputs):
            input_len = inputs['input_ids'][j].shape[0]
            generated = output[input_len:]
            summary = tokenizer.decode(generated, skip_special_tokens=True).strip()
            predictions.append(summary)
    
    return predictions


def compute_rouge(predictions, references):
    """Compute ROUGE scores with space-based Bangla tokenization."""
    from rouge_score import rouge_scorer
    
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=False,
        tokenizer=SpaceTokenizer()
    )
    
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(score[key].fmeasure)
    
    return {key: np.mean(vals) for key, vals in scores.items()}


def compute_bertscore(predictions, references):
    """Compute BERTScore for Bengali text."""
    try:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(
            predictions, references,
            lang="bn",
            verbose=True,
            batch_size=32,
        )
        return {
            'bertscore_precision': P.mean().item(),
            'bertscore_recall': R.mean().item(),
            'bertscore_f1': F1.mean().item(),
        }
    except ImportError:
        print("  WARNING: bert_score not installed. Skipping BERTScore.")
        print("  Install with: pip install bert_score")
        return {}


def compute_bleu(predictions, references):
    """Compute BLEU score with space-based tokenization for Bangla."""
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        
        # Tokenize using space (Bangla)
        refs_tokenized = [[ref.split()] for ref in references]  # list of [list of tokens]
        preds_tokenized = [pred.split() for pred in predictions]
        
        smoother = SmoothingFunction().method1
        
        bleu_1 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(1, 0, 0, 0), smoothing_function=smoother)
        bleu_2 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
        bleu_4 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)
        
        return {
            'bleu_1': bleu_1,
            'bleu_2': bleu_2,
            'bleu_4': bleu_4,
        }
    except ImportError:
        print("  WARNING: nltk not installed. Skipping BLEU.")
        print("  Install with: pip install nltk")
        return {}


def compute_semantic_similarity(predictions, references):
    """Compute semantic similarity using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        
        # Use a multilingual model that supports Bangla
        print("  Loading multilingual sentence-transformer...")
        st_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        print("  Encoding predictions...")
        pred_embeddings = st_model.encode(predictions, batch_size=64, show_progress_bar=True)
        print("  Encoding references...")
        ref_embeddings = st_model.encode(references, batch_size=64, show_progress_bar=True)
        
        # Cosine similarity per pair
        from numpy.linalg import norm
        similarities = []
        for p_emb, r_emb in zip(pred_embeddings, ref_embeddings):
            cos_sim = np.dot(p_emb, r_emb) / (norm(p_emb) * norm(r_emb) + 1e-8)
            similarities.append(float(cos_sim))
        
        return {
            'semantic_similarity_mean': np.mean(similarities),
            'semantic_similarity_std': np.std(similarities),
            'semantic_similarity_median': float(np.median(similarities)),
        }
    except ImportError:
        print("  WARNING: sentence-transformers not installed. Skipping Semantic Similarity.")
        print("  Install with: pip install sentence-transformers")
        return {}


def teacher_disagreement_analysis(test_data):
    """
    Analyze teacher disagreement patterns.
    Loads teacher outputs and computes JSD, entropy, agreement statistics.
    """
    print(f"\n{'='*80}")
    print("TEACHER DISAGREEMENT ANALYSIS")
    print(f"{'='*80}")
    
    teacher_32b_file = os.path.join(TEACHER_32B_OUTPUTS, "test.jsonl")
    teacher_14b_file = os.path.join(TEACHER_14B_OUTPUTS, "test.jsonl")
    
    if not os.path.exists(teacher_32b_file) or not os.path.exists(teacher_14b_file):
        print("  Teacher test outputs not found. Skipping analysis.")
        return {}
    
    # Load teacher outputs
    def load_jsonl(path):
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    
    t32b = load_jsonl(teacher_32b_file)
    t14b = load_jsonl(teacher_14b_file)
    
    min_len = min(len(t32b), len(t14b), len(test_data))
    
    # Compute per-sample disagreement metrics
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False, tokenizer=SpaceTokenizer())
    
    analysis = {
        'teacher_32b_avg_tokens': [],
        'teacher_14b_avg_tokens': [],
        'teacher_mutual_rouge': [],
        'teacher_32b_vs_gold_rouge': [],
        'teacher_14b_vs_gold_rouge': [],
    }
    
    for i in range(min_len):
        summary_32b = t32b[i].get('summary', '')
        summary_14b = t14b[i].get('summary', '')
        gold = test_data[i][DATASET_SUMMARY_KEY]
        
        analysis['teacher_32b_avg_tokens'].append(len(summary_32b.split()))
        analysis['teacher_14b_avg_tokens'].append(len(summary_14b.split()))
        
        # Teacher mutual agreement (ROUGE between their summaries)
        mutual_score = scorer.score(summary_32b, summary_14b)
        analysis['teacher_mutual_rouge'].append(mutual_score['rougeL'].fmeasure)
        
        # Teacher vs gold
        score_32b = scorer.score(gold, summary_32b)
        score_14b = scorer.score(gold, summary_14b)
        analysis['teacher_32b_vs_gold_rouge'].append(score_32b['rougeL'].fmeasure)
        analysis['teacher_14b_vs_gold_rouge'].append(score_14b['rougeL'].fmeasure)
    
    results = {}
    for k, v in analysis.items():
        results[k + '_mean'] = np.mean(v)
        results[k + '_std'] = np.std(v)
    
    print(f"  Teacher 32B avg output tokens: {results['teacher_32b_avg_tokens_mean']:.1f}")
    print(f"  Teacher 14B avg output tokens: {results['teacher_14b_avg_tokens_mean']:.1f}")
    print(f"  Teacher mutual ROUGE-L: {results['teacher_mutual_rouge_mean']:.4f}")
    print(f"  Teacher 32B vs Gold ROUGE-L: {results['teacher_32b_vs_gold_rouge_mean']:.4f}")
    print(f"  Teacher 14B vs Gold ROUGE-L: {results['teacher_14b_vs_gold_rouge_mean']:.4f}")
    
    return results


def evaluate_model(model_dir, run_analysis=False, quick_eval=False):
    """Full evaluation pipeline. If quick_eval=True, test on QUICK_EVAL_SAMPLES only."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    print(f"\n{'='*80}")
    print(f"EVALUATION")
    print(f"{'='*80}")
    print(f"Model: {model_dir}")
    
    # Detect if this is a LoRA model or full model
    is_lora = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
    
    if is_lora:
        print("  Detected LoRA adapter. Loading base + adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_dir)
        model = model.merge_and_unload()  # Merge LoRA weights
    else:
        print("  Loading full model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load test data
    test_data = load_test_data(quick_eval=quick_eval)
    references = [item[DATASET_SUMMARY_KEY] for item in test_data]
    
    # Generate summaries
    print("\nGenerating summaries...")
    predictions = generate_summaries(
        model, tokenizer, test_data,
        batch_size=EVAL_BATCH_SIZE,
        max_new_tokens=EVAL_MAX_LENGTH,
    )
    
    # ROUGE scores
    print("\nComputing ROUGE...")
    rouge_scores = compute_rouge(predictions, references)
    print(f"  ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {rouge_scores['rougeL']:.4f}")
    
    # BLEU scores
    print("\nComputing BLEU...")
    bleu_scores = compute_bleu(predictions, references)
    if bleu_scores:
        print(f"  BLEU-1: {bleu_scores['bleu_1']:.4f}")
        print(f"  BLEU-2: {bleu_scores['bleu_2']:.4f}")
        print(f"  BLEU-4: {bleu_scores['bleu_4']:.4f}")
    
    # BERTScore
    print("\nComputing BERTScore...")
    bert_scores = compute_bertscore(predictions, references)
    if bert_scores:
        print(f"  BERTScore F1: {bert_scores['bertscore_f1']:.4f}")
    
    # Semantic Similarity
    print("\nComputing Semantic Similarity...")
    sem_sim_scores = compute_semantic_similarity(predictions, references)
    if sem_sim_scores:
        print(f"  Semantic Similarity (mean): {sem_sim_scores['semantic_similarity_mean']:.4f}")
        print(f"  Semantic Similarity (std):  {sem_sim_scores['semantic_similarity_std']:.4f}")
    
    # Combine all results
    results = {
        "model_dir": model_dir,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "num_test_samples": len(test_data),
        **rouge_scores,
        **bleu_scores,
        **bert_scores,
        **sem_sim_scores,
    }
    
    # Teacher disagreement analysis
    if run_analysis:
        analysis_results = teacher_disagreement_analysis(test_data)
        results['teacher_analysis'] = analysis_results
    
    # Sample predictions
    print(f"\n{'='*80}")
    print("SAMPLE PREDICTIONS")
    print(f"{'='*80}")
    
    for i in range(min(5, len(predictions))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Input (first 150 chars): {test_data[i][DATASET_TEXT_KEY][:150]}...")
        print(f"Gold: {references[i][:200]}")
        print(f"Pred: {predictions[i][:200]}")
    
    # Save results
    output_parent = os.path.dirname(model_dir)
    results_file = os.path.join(output_parent, "eval_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_file}")
    
    # Save predictions
    preds_file = os.path.join(output_parent, "predictions.json")
    with open(preds_file, "w", encoding="utf-8") as f:
        json.dump([
            {"gold": ref, "prediction": pred}
            for ref, pred in zip(references, predictions)
        ], f, indent=2, ensure_ascii=False)
    print(f"Predictions saved to: {preds_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate student model")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to model directory")
    parser.add_argument("--analysis", action="store_true", help="Run teacher disagreement analysis")
    parser.add_argument("--quick", action="store_true", help=f"Quick eval on {QUICK_EVAL_SAMPLES} samples only")
    args = parser.parse_args()
    
    results = evaluate_model(args.model_dir, run_analysis=args.analysis, quick_eval=args.quick)
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif not isinstance(v, dict):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
