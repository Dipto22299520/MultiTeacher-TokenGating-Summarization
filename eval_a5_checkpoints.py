"""
Evaluate A5 checkpoints for Semantic Similarity only.
Picks the best checkpoint among the three available.
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
A5_DIR = "ablation_results_bansum/A5_full_pipeline_20260228_034939_batch16_backup"
BANSUM_FILE = "bansum_lte_1000_tokens.json"
SEED = 42
MAX_SAMPLES = 500  # Use subset for speed; set 0 for full test set
BATCH_SIZE = 16
MAX_INPUT_LEN = 512
MAX_OUTPUT_LEN = 150

CHECKPOINTS = ["checkpoint-7500", "checkpoint-13500", "checkpoint-15000"]


def load_bansum_test():
    """Load BanSum JSON and return the test split (matching 80/10/10 split)."""
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
    """Generate summaries for a list of texts."""
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
    """Compute mean cosine semantic similarity."""
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
    print("A5 Checkpoint Semantic Similarity Evaluation")
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

    # Evaluate each checkpoint
    results = {}
    for ckpt in CHECKPOINTS:
        ckpt_path = os.path.join(A5_DIR, ckpt)
        if not os.path.exists(ckpt_path):
            print(f"\n  SKIP: {ckpt} (not found)")
            continue

        print(f"\n{'─'*70}")
        print(f"Evaluating: {ckpt}")
        print(f"{'─'*70}")

        preds = generate_summaries(ckpt_path, texts)

        # Filter empty
        valid = [(p, r) for p, r in zip(preds, references) if p.strip()]
        if len(valid) < len(preds):
            print(f"  WARNING: {len(preds) - len(valid)} empty predictions filtered")
        preds_clean = [v[0] for v in valid]
        refs_clean = [v[1] for v in valid]

        sem_sim = compute_semantic_similarity(preds_clean, refs_clean, sem_model)
        results[ckpt] = sem_sim
        print(f"  Semantic Similarity: {sem_sim:.4f}  ({len(preds_clean)} samples)")

    # Summary
    print(f"\n\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    for ckpt, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ckpt:<20} -> Semantic Similarity = {score:.4f}")

    best_ckpt = max(results, key=results.get)
    best_score = results[best_ckpt]
    print(f"\n  ** BEST: {best_ckpt}  (score = {best_score:.4f}) **")

    # Rename best to best_model and delete others
    best_src = os.path.join(A5_DIR, best_ckpt)
    best_dst = os.path.join(A5_DIR, "best_model")

    if os.path.exists(best_dst):
        print(f"\n  Removing existing best_model...")
        shutil.rmtree(best_dst)

    print(f"  Renaming {best_ckpt} -> best_model")
    os.rename(best_src, best_dst)

    # Delete the other checkpoints
    for ckpt in CHECKPOINTS:
        ckpt_path = os.path.join(A5_DIR, ckpt)
        if os.path.exists(ckpt_path):
            print(f"  Deleting {ckpt}...")
            shutil.rmtree(ckpt_path)

    # Also clean up optimizer/scheduler/rng from best_model (not needed for inference)
    for fname in ["optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin", "trainer_state.json"]:
        fpath = os.path.join(best_dst, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"  Cleaned up {fname} from best_model")

    # Save evaluation results
    eval_out = {
        "evaluation": "A5_checkpoint_semantic_similarity",
        "test_samples": len(texts),
        "results": {k: round(v, 4) for k, v in results.items()},
        "best_checkpoint": best_ckpt,
        "best_score": round(best_score, 4),
    }
    eval_path = os.path.join(A5_DIR, "checkpoint_eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_out, f, indent=2, ensure_ascii=False)
    print(f"\n  Evaluation results saved to {eval_path}")

    print(f"\n  Done! best_model is now {best_ckpt} at {best_dst}")


if __name__ == "__main__":
    main()
