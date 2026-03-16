"""
Evaluate Ablation Models (BanSum) — BLEU, BERTScore, Semantic Similarity
=========================================================================
Generates summaries from each BanSum ablation best_model on the BanSum test set,
then computes:
  - BLEU (sacrebleu)
  - BERTScore (F1)
  - Semantic Similarity (sentence-transformers cosine similarity)

Usage:
  python evaluate_ablation_metrics_bansum.py
  python evaluate_ablation_metrics_bansum.py --max_samples 500
  python evaluate_ablation_metrics_bansum.py --include_teacher
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu
from bert_score import score as bert_score_fn
from sentence_transformers import SentenceTransformer, util

# ============================================================================
# Config (BanSum-specific)
# ============================================================================

ABLATION_DIR = "ablation_results_bansum"
TEACHER_DIR = "banglat5_bansum_20260218_213532/final_model"
BANSUM_FILE = "bansum_lte_1000_tokens.json"
SEED = 42

CONFIGS = [
    "A1_baseline",
    "A2_single_kd",
    "A3_multi_teacher",
    "A4_adaptive_temp",
    "A5_full_pipeline",
]

DESCRIPTIONS = {
    "A1_baseline": "Baseline: Fine-tune student alone (no distillation)",
    "A2_single_kd": "Single-teacher logit-level KD (BanglaT5-BanSum, fixed temperature)",
    "A3_multi_teacher": "Multi-teacher: Logit KD + pseudo-label augmentation from mT5-BanSum",
    "A4_adaptive_temp": "Multi-teacher + confidence-adaptive temperature",
    "A5_full_pipeline": "Full: Multi-teacher + adaptive temp + intermediate matching",
}


def load_bansum_test():
    """Load BanSum JSON and return the test split (matching 80/10/10 split)."""
    print(f"  Loading BanSum data from {BANSUM_FILE}...")
    with open(BANSUM_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame([{"text": item["main"], "summary": item["sum2"]} for item in data])
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    total = len(df)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_df = df[train_size + val_size:]
    print(f"  Total: {total}, Test split: {len(test_df)}")
    return test_df


def find_best_model_dir(config_name):
    """Find the best_model directory for a given ablation config."""
    if not os.path.exists(ABLATION_DIR):
        return None
    matching = sorted(
        [d for d in os.listdir(ABLATION_DIR) if d.startswith(config_name + "_")],
        reverse=True,
    )
    for run_dir in matching:
        best = os.path.join(ABLATION_DIR, run_dir, "best_model")
        if os.path.exists(best):
            return best
    return None


def generate_summaries(model_path, texts, batch_size=16, max_input_len=512, max_output_len=150):
    """Generate summaries for a list of texts using a seq2seq model."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    all_summaries = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating", leave=False):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=max_input_len,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_output_len,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_summaries.extend(decoded)

    del model
    torch.cuda.empty_cache()
    return all_summaries


def compute_bleu(predictions, references):
    """Compute corpus-level BLEU score."""
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score


def compute_bertscore(predictions, references):
    """Compute BERTScore (P, R, F1) using multilingual model."""
    P, R, F1 = bert_score_fn(
        predictions,
        references,
        lang="bn",
        verbose=False,
        batch_size=32,
    )
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
    }


def compute_semantic_similarity(predictions, references, sem_model):
    """Compute mean cosine semantic similarity using sentence-transformers."""
    pred_embeddings = sem_model.encode(predictions, batch_size=64, show_progress_bar=False)
    ref_embeddings = sem_model.encode(references, batch_size=64, show_progress_bar=False)

    similarities = []
    for p_emb, r_emb in zip(pred_embeddings, ref_embeddings):
        sim = util.cos_sim(
            torch.tensor(p_emb).unsqueeze(0),
            torch.tensor(r_emb).unsqueeze(0),
        ).item()
        similarities.append(sim)

    return float(np.mean(similarities))


def main():
    parser = argparse.ArgumentParser(description="Evaluate BanSum ablation models with BLEU/BERTScore/Semantic Similarity")
    parser.add_argument("--max_samples", type=int, default=0, help="Max test samples (0=all)")
    parser.add_argument("--batch_size", type=int, default=16, help="Generation batch size")
    parser.add_argument("--include_teacher", action="store_true", help="Also evaluate the BanSum teacher model")
    args = parser.parse_args()

    # Load test data from BanSum
    print("Loading BanSum test data...")
    test_df = load_bansum_test()
    if args.max_samples > 0:
        test_df = test_df.head(args.max_samples)
    texts = test_df["text"].tolist()
    references = test_df["summary"].tolist()
    print(f"  Test samples: {len(texts)}")

    # Load semantic similarity model once
    print("\nLoading sentence-transformer model for semantic similarity...")
    sem_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("  Loaded paraphrase-multilingual-MiniLM-L12-v2")

    # Models to evaluate
    models_to_eval = []
    for cfg in CONFIGS:
        model_dir = find_best_model_dir(cfg)
        if model_dir:
            models_to_eval.append((cfg, model_dir, DESCRIPTIONS.get(cfg, "")))
        else:
            print(f"  WARNING: No best_model found for {cfg}, skipping.")

    if args.include_teacher and os.path.exists(TEACHER_DIR):
        models_to_eval.append(("Teacher_BanglaT5_BanSum", TEACHER_DIR, "Teacher: BanglaT5 fine-tuned on BanSum"))

    all_results = {}

    for cfg_name, model_dir, desc in models_to_eval:
        print(f"\n{'='*80}")
        print(f"EVALUATING: {cfg_name}")
        print(f"  Model: {model_dir}")
        print(f"{'='*80}")

        # Generate summaries
        predictions = generate_summaries(model_dir, texts, batch_size=args.batch_size)

        # Filter out empty predictions
        valid = [(p, r) for p, r in zip(predictions, references) if p.strip()]
        if len(valid) < len(predictions):
            print(f"  WARNING: {len(predictions) - len(valid)} empty predictions filtered out")
        preds_clean = [v[0] for v in valid]
        refs_clean = [v[1] for v in valid]

        # BLEU
        print("  Computing BLEU...")
        bleu = compute_bleu(preds_clean, refs_clean)
        print(f"    BLEU: {bleu:.2f}")

        # BERTScore
        print("  Computing BERTScore...")
        bertscore = compute_bertscore(preds_clean, refs_clean)
        print(f"    BERTScore F1: {bertscore['f1']:.4f}")

        # Semantic Similarity
        print("  Computing Semantic Similarity...")
        sem_sim = compute_semantic_similarity(preds_clean, refs_clean, sem_model)
        print(f"    Semantic Similarity: {sem_sim:.4f}")

        all_results[cfg_name] = {
            "description": desc,
            "bleu": round(bleu, 2),
            "bertscore_precision": round(bertscore["precision"], 4),
            "bertscore_recall": round(bertscore["recall"], 4),
            "bertscore_f1": round(bertscore["f1"], 4),
            "semantic_similarity": round(sem_sim, 4),
            "num_samples": len(preds_clean),
        }

    # Print comparison table
    print(f"\n\n{'='*110}")
    print("ABLATION STUDY (BANSUM) — BLEU / BERTScore / Semantic Similarity")
    print(f"{'='*110}")
    header = (
        f"{'Config':<28} {'BLEU':>7} {'BERT-P':>8} {'BERT-R':>8} {'BERT-F1':>8} "
        f"{'SemSim':>8}  Description"
    )
    print(header)
    print("-" * 110)

    baseline_sem = None
    teacher_key = "Teacher_BanglaT5_BanSum"
    for cfg_name in CONFIGS + ([teacher_key] if args.include_teacher else []):
        r = all_results.get(cfg_name)
        if r is None:
            print(f"{cfg_name:<28}  {'—':>7} {'—':>8} {'—':>8} {'—':>8} {'—':>8}  (not evaluated)")
            continue

        if baseline_sem is None:
            baseline_sem = r["semantic_similarity"]

        delta_sem = ""
        if cfg_name != "A1_baseline" and cfg_name != teacher_key and baseline_sem:
            diff = r["semantic_similarity"] - baseline_sem
            delta_sem = f" ({diff:+.4f})"

        print(
            f"{cfg_name:<28} {r['bleu']:>7.2f} {r['bertscore_precision']:>8.4f} "
            f"{r['bertscore_recall']:>8.4f} {r['bertscore_f1']:>8.4f} "
            f"{r['semantic_similarity']:>8.4f}{delta_sem:<10} {r['description']}"
        )

    print("-" * 110)

    # Best config by each metric
    student_results = {k: v for k, v in all_results.items() if k.startswith("A")}
    if student_results:
        best = max(student_results.items(), key=lambda x: x[1]["semantic_similarity"])
        print(f"\nBest config (Semantic Similarity): {best[0]} = {best[1]['semantic_similarity']:.4f}")
        best_bleu = max(student_results.items(), key=lambda x: x[1]["bleu"])
        print(f"Best config (BLEU):                {best_bleu[0]} = {best_bleu[1]['bleu']:.2f}")
        best_bert = max(student_results.items(), key=lambda x: x[1]["bertscore_f1"])
        print(f"Best config (BERTScore F1):         {best_bert[0]} = {best_bert[1]['bertscore_f1']:.4f}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "bansum_lte_1000",
        "num_test_samples": len(texts),
        "metrics": ["BLEU", "BERTScore", "Semantic Similarity"],
        "semantic_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "results": all_results,
    }
    os.makedirs(ABLATION_DIR, exist_ok=True)
    out_path = os.path.join(ABLATION_DIR, "ablation_extended_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
