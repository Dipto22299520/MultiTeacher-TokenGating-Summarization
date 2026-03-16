"""
Evaluate A1-A4 checkpoints for Semantic Similarity.
For each config, picks the best checkpoint and renames it to best_model.
"""

import os
import json
import shutil
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# ============================================================================
# Config
# ============================================================================
ABLATION_DIR = "ablation_results_bansum"
BANSUM_FILE = "bansum_lte_1000_tokens.json"
SEED = 42
MAX_SAMPLES = 500  # Use subset for speed; set 0 for full test set
BATCH_SIZE = 16
MAX_INPUT_LEN = 512
MAX_OUTPUT_LEN = 150

CONFIGS = {
    "A1_baseline_20260224_121256": ["checkpoint-15000", "checkpoint-16500", "checkpoint-17650"],
    "A2_single_kd_20260225_120221": ["checkpoint-12000", "checkpoint-16500", "checkpoint-17650"],
    "A3_multi_teacher_20260226_095657": ["checkpoint-12000", "checkpoint-16500", "checkpoint-17650"],
    "A4_adaptive_temp_20260227_192419": ["checkpoint-15000", "checkpoint-16500", "checkpoint-17650"],
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
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_summaries.extend(decoded)

    del model
    torch.cuda.empty_cache()
    return all_summaries


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
    print("=" * 70)
    print("A1-A4 Checkpoint Semantic Similarity Evaluation")
    print("=" * 70)

    # Load test data
    print("\nLoading BanSum test data...")
    test_df = load_bansum_test()
    if MAX_SAMPLES > 0:
        test_df = test_df.head(MAX_SAMPLES)
    texts = test_df["text"].tolist()
    references = test_df["summary"].tolist()
    print(f"  Test samples: {len(texts)}")

    # Load semantic model once
    print("\nLoading sentence-transformer model...")
    sem_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    all_config_results = {}

    for config_dir, checkpoints in CONFIGS.items():
        config_name = config_dir.split("_2026")[0]  # e.g. A1_baseline
        config_path = os.path.join(ABLATION_DIR, config_dir)

        print(f"\n\n{'='*70}")
        print(f"CONFIG: {config_name} ({config_dir})")
        print(f"{'='*70}")

        results = {}

        # Also evaluate existing best_model for comparison
        best_model_path = os.path.join(config_path, "best_model")
        eval_targets = []
        if os.path.exists(best_model_path):
            eval_targets.append(("best_model (current)", best_model_path))
        for ckpt in checkpoints:
            ckpt_path = os.path.join(config_path, ckpt)
            if os.path.exists(ckpt_path):
                eval_targets.append((ckpt, ckpt_path))

        for name, path in eval_targets:
            print(f"\n{'─'*70}")
            print(f"Evaluating: {name}")
            print(f"{'─'*70}")

            preds = generate_summaries(path, texts)
            valid = [(p, r) for p, r in zip(preds, references) if p.strip()]
            if len(valid) < len(preds):
                print(f"  WARNING: {len(preds) - len(valid)} empty predictions filtered")
            preds_clean = [v[0] for v in valid]
            refs_clean = [v[1] for v in valid]

            sem_sim = compute_semantic_similarity(preds_clean, refs_clean, sem_model)
            results[name] = sem_sim
            print(f"  Semantic Similarity: {sem_sim:.4f}  ({len(preds_clean)} samples)")

        # Summary for this config
        print(f"\n  RESULTS for {config_name}:")
        for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"    {name:<25} -> {score:.4f}")

        # Find best among checkpoints only (not existing best_model)
        ckpt_results = {k: v for k, v in results.items() if k != "best_model (current)"}
        best_ckpt = max(ckpt_results, key=ckpt_results.get)
        best_score = ckpt_results[best_ckpt]
        print(f"\n  ** BEST CHECKPOINT: {best_ckpt}  (score = {best_score:.4f}) **")

        # Remove old best_model
        if os.path.exists(best_model_path):
            print(f"  Removing old best_model...")
            shutil.rmtree(best_model_path)

        # Rename best checkpoint to best_model
        best_src = os.path.join(config_path, best_ckpt)
        print(f"  Renaming {best_ckpt} -> best_model")
        os.rename(best_src, best_model_path)

        # Delete other checkpoints
        for ckpt in checkpoints:
            ckpt_path = os.path.join(config_path, ckpt)
            if os.path.exists(ckpt_path):
                print(f"  Deleting {ckpt}...")
                shutil.rmtree(ckpt_path)

        # Clean up training artifacts from best_model
        for fname in ["optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin", "trainer_state.json"]:
            fpath = os.path.join(best_model_path, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
                print(f"  Cleaned up {fname}")

        all_config_results[config_name] = {
            "all_scores": {k: round(v, 4) for k, v in results.items()},
            "best_checkpoint": best_ckpt,
            "best_score": round(best_score, 4),
        }

    # Final summary
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY — ALL CONFIGS")
    print(f"{'='*70}")
    for cfg, info in all_config_results.items():
        print(f"  {cfg:<25} -> best = {info['best_checkpoint']:<20} (SemSim = {info['best_score']:.4f})")

    # Save results
    out_path = os.path.join(ABLATION_DIR, "a1_a4_checkpoint_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"test_samples": len(texts), "configs": all_config_results}, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
