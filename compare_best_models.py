"""
Compare all 5 ablation best_models: ROUGE-1/2/L, BLEU, BERTScore, Semantic Similarity
======================================================================================
Evaluates on the BanSum test set and saves a combined JSON.
"""

import os
import json
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu
from bert_score import score as bert_score_fn
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
import pandas as pd

# ============================================================================
# Config
# ============================================================================
ABLATION_DIR = "ablation_results_bansum"
BANSUM_FILE = "bansum_lte_1000_tokens.json"
SEED = 42
MAX_SAMPLES = 0    # 0 = use full test set
BATCH_SIZE = 4
MAX_INPUT_LEN = 512
MAX_OUTPUT_LEN = 150

MODELS = {
    "A1_baseline": os.path.join(ABLATION_DIR, "A1_baseline_20260224_121256", "best_model"),
    "A2_single_kd": os.path.join(ABLATION_DIR, "A2_single_kd_20260225_120221", "best_model"),
    "A3_multi_teacher": os.path.join(ABLATION_DIR, "A3_multi_teacher_20260226_095657", "best_model"),
    "A4_adaptive_temp": os.path.join(ABLATION_DIR, "A4_adaptive_temp_20260227_192419", "best_model"),
    "A5_full_pipeline": os.path.join(ABLATION_DIR, "A5_full_pipeline_20260228_034939_batch16_backup", "best_model"),
}

DESCRIPTIONS = {
    "A1_baseline": "Baseline: Fine-tune student alone (no distillation)",
    "A2_single_kd": "Single-teacher logit-level KD (BanglaT5-BanSum, fixed temp)",
    "A3_multi_teacher": "Multi-teacher: Logit KD + pseudo-label augmentation",
    "A4_adaptive_temp": "Multi-teacher + confidence-adaptive temperature",
    "A5_full_pipeline": "Full: Multi-teacher + adaptive temp + intermediate matching",
}

# Semantic similarity scores from checkpoint evaluation
SEMSIM_SCORES = {
    "A1_baseline": 0.7377,
    "A2_single_kd": 0.7601,
    "A3_multi_teacher": 0.7646,
    "A4_adaptive_temp": 0.7247,
    "A5_full_pipeline": 0.7442,
}


def load_bansum_test():
    with open(BANSUM_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame([{"text": item["main"], "summary": item["sum2"]} for item in data])
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    total = len(df)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_df = df[train_size + val_size:]
    return test_df


def generate_summaries(model_path, texts):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    all_summaries = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Generating", leave=False):
        batch_texts = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_OUTPUT_LEN,
                num_beams=2,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_summaries.extend(decoded)

    del model
    torch.cuda.empty_cache()
    return all_summaries


class SpaceTokenizer:
    """Space-based tokenizer for Bangla ROUGE evaluation."""
    def tokenize(self, text):
        return text.split()


def compute_rouge(predictions, references):
    """Compute ROUGE-1, ROUGE-2, ROUGE-L (F1) scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False, tokenizer=SpaceTokenizer())
    r1_scores, r2_scores, rl_scores = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_scores.append(scores["rouge1"].fmeasure)
        r2_scores.append(scores["rouge2"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)
    return {
        "rouge1": float(np.mean(r1_scores)),
        "rouge2": float(np.mean(r2_scores)),
        "rougeL": float(np.mean(rl_scores)),
    }


def compute_bleu(predictions, references):
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score


def compute_bertscore(predictions, references):
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
    pred_emb = sem_model.encode(predictions, batch_size=64, show_progress_bar=False)
    ref_emb = sem_model.encode(references, batch_size=64, show_progress_bar=False)
    sims = []
    for p, r in zip(pred_emb, ref_emb):
        sim = util.cos_sim(
            torch.tensor(p).unsqueeze(0),
            torch.tensor(r).unsqueeze(0),
        ).item()
        sims.append(sim)
    return float(np.mean(sims))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="*", default=[], help="Configs to skip (e.g. --skip A1_baseline)")
    args = parser.parse_args()

    PARTIAL_FILE = os.path.join(ABLATION_DIR, "ablation_full_metrics_comparison_partial.json")

    print("=" * 90)
    print("ABLATION STUDY — Full Metrics Comparison (A1-A5)")
    print("=" * 90)

    # Load test data
    print("\nLoading BanSum test data...")
    test_df = load_bansum_test()
    if MAX_SAMPLES > 0:
        test_df = test_df.head(MAX_SAMPLES)
    texts = test_df["text"].tolist()
    references = test_df["summary"].tolist()
    print(f"  Test samples: {len(texts)}")

    # Load semantic model
    print("\nLoading sentence-transformer model...")
    sem_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Load partial results if they exist
    all_results = {}
    if os.path.exists(PARTIAL_FILE):
        with open(PARTIAL_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
            all_results = saved.get("results", {})
            print(f"  Loaded partial results for: {list(all_results.keys())}")

    for config_name, model_path in MODELS.items():
        if config_name in args.skip:
            print(f"\n  SKIP (--skip): {config_name}")
            continue

        if config_name in all_results:
            print(f"\n  SKIP (already done): {config_name}")
            continue

        if not os.path.exists(model_path):
            print(f"\n  SKIP: {config_name} — best_model not found at {model_path}")
            continue

        print(f"\n{'='*90}")
        print(f"  EVALUATING: {config_name}")
        print(f"  Model: {model_path}")
        print(f"{'='*90}")

        # Generate summaries
        preds = generate_summaries(model_path, texts)

        # Filter empty
        valid = [(p, r) for p, r in zip(preds, references) if p.strip()]
        if len(valid) < len(preds):
            print(f"  WARNING: {len(preds) - len(valid)} empty predictions filtered")
        preds_clean = [v[0] for v in valid]
        refs_clean = [v[1] for v in valid]

        # ROUGE
        print("  Computing ROUGE...")
        rouge = compute_rouge(preds_clean, refs_clean)
        print(f"    ROUGE-1: {rouge['rouge1']:.4f}")
        print(f"    ROUGE-2: {rouge['rouge2']:.4f}")
        print(f"    ROUGE-L: {rouge['rougeL']:.4f}")

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

        all_results[config_name] = {
            "description": DESCRIPTIONS.get(config_name, ""),
            "model_path": model_path,
            "num_samples": len(preds_clean),
            "rouge1": round(rouge["rouge1"], 4),
            "rouge2": round(rouge["rouge2"], 4),
            "rougeL": round(rouge["rougeL"], 4),
            "bleu": round(bleu, 2),
            "bertscore_precision": round(bertscore["precision"], 4),
            "bertscore_recall": round(bertscore["recall"], 4),
            "bertscore_f1": round(bertscore["f1"], 4),
            "semantic_similarity": round(sem_sim, 4),
        }

        # Save partial results after each model
        partial_out = {
            "timestamp": datetime.now().isoformat(),
            "dataset": "bansum_lte_1000",
            "split": "test",
            "num_test_samples": len(texts),
            "results": all_results,
        }
        with open(PARTIAL_FILE, "w", encoding="utf-8") as f:
            json.dump(partial_out, f, indent=2, ensure_ascii=False)
        print(f"  -> Saved partial results ({len(all_results)}/{len(MODELS)} done)")

    # ============================================================================
    # Print comparison table
    # ============================================================================
    print(f"\n\n{'='*120}")
    print("ABLATION STUDY — FULL METRICS COMPARISON")
    print(f"{'='*120}")
    header = (
        f"{'Config':<22} {'R-1':>6} {'R-2':>6} {'R-L':>6} {'BLEU':>7} "
        f"{'BERT-P':>8} {'BERT-R':>8} {'BERT-F1':>8} {'SemSim':>8}"
    )
    print(header)
    print("-" * 120)

    for cfg in MODELS:
        r = all_results.get(cfg)
        if r is None:
            print(f"  {cfg:<22}  (not evaluated)")
            continue
        print(
            f"  {cfg:<22} {r['rouge1']:>6.4f} {r['rouge2']:>6.4f} {r['rougeL']:>6.4f} "
            f"{r['bleu']:>7.2f} {r['bertscore_precision']:>8.4f} {r['bertscore_recall']:>8.4f} "
            f"{r['bertscore_f1']:>8.4f} {r['semantic_similarity']:>8.4f}"
        )

    print("-" * 120)

    # Best by each metric
    if all_results:
        metrics = [
            ("ROUGE-1", "rouge1"),
            ("ROUGE-2", "rouge2"),
            ("ROUGE-L", "rougeL"),
            ("BLEU", "bleu"),
            ("BERTScore F1", "bertscore_f1"),
            ("Semantic Sim", "semantic_similarity"),
        ]
        print("\nBest config per metric:")
        for label, key in metrics:
            best = max(all_results.items(), key=lambda x: x[1][key])
            print(f"  {label:<15} -> {best[0]:<22} = {best[1][key]}")

    # Save JSON
    output = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "bansum_lte_1000",
        "split": "test",
        "num_test_samples": len(texts),
        "metrics": ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore", "Semantic Similarity"],
        "semantic_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "results": all_results,
    }
    out_path = os.path.join(ABLATION_DIR, "ablation_full_metrics_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
