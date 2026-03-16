"""
Evaluate BanSum teacher checkpoints with BLEU, BERTScore, Semantic Similarity.

Usage:
  python evaluate_teachers_bansum.py
  python evaluate_teachers_bansum.py --batch_size 16 --max_samples 0
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

BANSUM_FILE = "bansum_lte_1000_tokens.json"
SEED = 42

TEACHER_MODELS = {
    "banglaT5_bansum": {
        "path": "banglat5_bansum_20260218_213532/final_model",
        "description": "BanglaT5 fine-tuned on BanSum (checkpoint-14000)",
    }
}


def load_bansum_test():
    """Load BanSum JSON and return the test split (matching 80/10/10 split)."""
    print(f"Loading BanSum data from {BANSUM_FILE}...")
    with open(BANSUM_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame([{"text": item["main"], "summary": item["sum2"]} for item in data])
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    total = len(df)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_df = df[train_size + val_size:]
    print(f"Total: {total}, Test split: {len(test_df)}")
    return test_df


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
    parser = argparse.ArgumentParser(description="Evaluate BanSum teacher checkpoints with BLEU/BERTScore/Semantic Similarity")
    parser.add_argument("--max_samples", type=int, default=0, help="Max test samples (0=all)")
    parser.add_argument("--batch_size", type=int, default=16, help="Generation batch size")
    parser.add_argument("--out", type=str, default="teacher_metrics_bansum.json", help="Output JSON path")
    args = parser.parse_args()

    test_df = load_bansum_test()
    if args.max_samples > 0:
        test_df = test_df.head(args.max_samples)
    texts = test_df["text"].tolist()
    references = test_df["summary"].tolist()
    print(f"Test samples: {len(texts)}")

    print("Loading sentence-transformer model for semantic similarity...")
    sem_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("Loaded paraphrase-multilingual-MiniLM-L12-v2")

    all_results = {}

    for key, info in TEACHER_MODELS.items():
        model_dir = info["path"]
        if not os.path.exists(model_dir):
            print(f"WARNING: Missing model directory: {model_dir}, skipping.")
            continue

        print(f"\n{'='*80}")
        print(f"EVALUATING: {key}")
        print(f"Model: {model_dir}")
        print(f"{'='*80}")

        predictions = generate_summaries(model_dir, texts, batch_size=args.batch_size)

        valid = [(p, r) for p, r in zip(predictions, references) if p.strip()]
        if len(valid) < len(predictions):
            print(f"WARNING: {len(predictions) - len(valid)} empty predictions filtered out")
        preds_clean = [v[0] for v in valid]
        refs_clean = [v[1] for v in valid]

        print("Computing BLEU...")
        bleu = compute_bleu(preds_clean, refs_clean)
        print(f"BLEU: {bleu:.2f}")

        print("Computing BERTScore...")
        bertscore = compute_bertscore(preds_clean, refs_clean)
        print(f"BERTScore F1: {bertscore['f1']:.4f}")

        print("Computing Semantic Similarity...")
        sem_sim = compute_semantic_similarity(preds_clean, refs_clean, sem_model)
        print(f"Semantic Similarity: {sem_sim:.4f}")

        all_results[key] = {
            "description": info["description"],
            "model_path": model_dir,
            "bleu": round(bleu, 2),
            "bertscore_precision": round(bertscore["precision"], 4),
            "bertscore_recall": round(bertscore["recall"], 4),
            "bertscore_f1": round(bertscore["f1"], 4),
            "semantic_similarity": round(sem_sim, 4),
            "num_samples": len(preds_clean),
        }

    output = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "bansum_lte_1000",
        "split": "test",
        "num_test_samples": len(texts),
        "metrics": ["BLEU", "BERTScore", "Semantic Similarity"],
        "semantic_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "results": all_results,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=True)

    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
