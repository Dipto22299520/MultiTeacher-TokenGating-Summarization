"""
Re-evaluate each student model against ground-truth summaries (not teacher labels).
Updates training_results.json with correct student ROUGE scores and honest retention.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

# ============================================================================
# Config
# ============================================================================

LANGUAGES = ["amharic", "hausa", "nepali", "pashto"]

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256
BATCH_SIZE = 16
USE_PREFIX = True

RESULTS_FILE = "./training_results.json"


class SpaceTokenizer:
    def tokenize(self, text):
        return text.split()


# ============================================================================
# Helpers
# ============================================================================

def find_latest_dir(base, pattern):
    matches = glob.glob(os.path.join(base, pattern))
    return max(matches, key=os.path.getmtime) if matches else None


def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=SpaceTokenizer()
    )
    r1, r2, rL = [], [], []
    for pred, ref in zip(predictions, references):
        pred = pred.strip() or "."
        ref = ref.strip() or "."
        s = scorer.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
    return {
        "rouge1": float(np.mean(r1)),
        "rouge2": float(np.mean(r2)),
        "rougeL": float(np.mean(rL)),
    }


def evaluate_student(model_dir, test_csv):
    """Run student model on test set and return ROUGE vs ground truth."""
    print(f"  Loading student model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    df = pd.read_csv(test_csv)
    print(f"  Test samples: {len(df)}")

    predictions = []
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="  Generating"):
        batch_texts = df["text"].iloc[i:i + BATCH_SIZE].tolist()

        if USE_PREFIX:
            batch_texts = ["summarize: " + t for t in batch_texts]

        inputs = tokenizer(
            batch_texts,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_TARGET_LENGTH,
                num_beams=4,
                early_stopping=True
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)

    torch.cuda.empty_cache()

    references = df["summary"].tolist()
    scores = compute_rouge(predictions, references)
    print(f"  ROUGE-1: {scores['rouge1']:.4f}  ROUGE-2: {scores['rouge2']:.4f}  ROUGE-L: {scores['rougeL']:.4f}")
    return scores


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("RE-EVALUATING STUDENTS AGAINST GROUND TRUTH")
    print("=" * 80)

    # Load existing results (teacher scores are correct)
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    for lang in LANGUAGES:
        print(f"\n{'=' * 70}")
        print(f"LANGUAGE: {lang.upper()}")
        print(f"{'=' * 70}")

        student_dir = find_latest_dir("./students", f"{lang}_student_fast_*")
        if not student_dir:
            print(f"  ERROR: No student found for {lang}, skipping.")
            continue

        # Ground-truth test file (NOT the teacher-label version)
        test_csv = f"./preprocessed_data/{lang}/test.csv"
        if not os.path.exists(test_csv):
            print(f"  ERROR: Ground-truth test file not found: {test_csv}")
            continue

        student_scores = evaluate_student(student_dir, test_csv)

        teacher_scores = results[lang]["teacher"]

        retention = {
            "rouge1": round(student_scores["rouge1"] / teacher_scores["rouge1"] * 100, 2),
            "rouge2": round(student_scores["rouge2"] / teacher_scores["rouge2"] * 100, 2),
            "rougeL": round(student_scores["rougeL"] / teacher_scores["rougeL"] * 100, 2),
        }

        results[lang]["student"] = {
            "rouge1": round(student_scores["rouge1"], 6),
            "rouge2": round(student_scores["rouge2"], 6),
            "rougeL": round(student_scores["rougeL"], 6),
        }
        results[lang]["retention"] = retention
        results[lang]["student_eval_note"] = "evaluated against ground-truth summaries"

        print(f"  Retention — R1: {retention['rouge1']}%  R2: {retention['rouge2']}%  RL: {retention['rougeL']}%")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"{'Language':<12} {'T-R1':>7} {'T-R2':>7} {'T-RL':>7}  {'S-R1':>7} {'S-R2':>7} {'S-RL':>7}  {'Ret-RL':>8}")
    print("-" * 80)
    for lang in LANGUAGES:
        t = results[lang]["teacher"]
        s = results[lang]["student"]
        r = results[lang]["retention"]
        print(f"{lang:<12} {t['rouge1']:>7.4f} {t['rouge2']:>7.4f} {t['rougeL']:>7.4f}  "
              f"{s['rouge1']:>7.4f} {s['rouge2']:>7.4f} {s['rougeL']:>7.4f}  {r['rougeL']:>7.2f}%")
    print("=" * 80)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
