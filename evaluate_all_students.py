"""
Evaluate all student models: BLEU, BERTScore, Semantic Similarity
Collects existing ROUGE from saved results, adds new metrics, builds final JSON.
"""

import json
import os
import glob
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

STUDENTS = {
    "hindi": {
        "model": "./students/hindi_student_fast_20260306_212437/checkpoint-1233",
        "test": "./preprocessed_data/hindi_finetuned_teacher_labels/test.csv",
        "teacher_results": "./teachers/hindi_teacher_20260306_181347/test_results.json",
        "student_results": "./students/hindi_student_fast_20260306_212437/test_results.json",
    },
    "urdu": {
        "model": "./students/urdu_student_fast_20260306_230834/checkpoint-1395",
        "test": "./preprocessed_data/urdu_finetuned_teacher_labels/test.csv",
        "teacher_results": "./teachers/urdu_teacher_20260306_213245/test_results.json",
        "student_results": "./students/urdu_student_fast_20260306_230834/test_results.json",
    },
    "russian": {
        "model": "./students/russian_student_fast_20260307_013515/checkpoint-1488",
        "test": "./preprocessed_data/russian_finetuned_teacher_labels/test.csv",
        "teacher_results": "./teachers/russian_teacher_20260307_000405/test_results.json",
        "student_results": "./students/russian_student_fast_20260307_013515/test_results.json",
    },
    "portuguese": {
        "model": "./students/portuguese_student_fast_20260307_031722/checkpoint-1323",
        "test": "./preprocessed_data/portuguese_finetuned_teacher_labels/test.csv",
        "teacher_results": "./teachers/portuguese_teacher_20260307_014259/test_results.json",
        "student_results": "./students/portuguese_student_fast_20260307_031722/test_results.json",
    },
    "persian": {
        "model": "./students/persian_student_fast_20260307_041138/checkpoint-726",
        "test": "./preprocessed_data/persian_finetuned_teacher_labels/test.csv",
        "teacher_results": "./teachers/persian_teacher_20260307_032340/test_results.json",
        "student_results": "./students/persian_student_fast_20260307_041138/test_results.json",
    },
    # -------------------------------------------------------------------------
    # New low-resource languages — fill in the timestamped paths after training
    # -------------------------------------------------------------------------
    "nepali": {
        "model": "./students/nepali_student_fast_TIMESTAMP/checkpoint-XXXX",   # update after training
        "test": "./preprocessed_data/nepali_finetuned_teacher_labels/test.csv",
        "teacher_results": "./teachers/nepali_teacher_TIMESTAMP/test_results.json",
        "student_results": "./students/nepali_student_fast_TIMESTAMP/test_results.json",
    },
    "amharic": {
        "model": "./students/amharic_student_fast_TIMESTAMP/checkpoint-XXXX",
        "test": "./preprocessed_data/amharic_finetuned_teacher_labels/test.csv",
        "teacher_results": "./teachers/amharic_teacher_TIMESTAMP/test_results.json",
        "student_results": "./students/amharic_student_fast_TIMESTAMP/test_results.json",
    },
    "pashto": {
        "model": "./students/pashto_student_fast_TIMESTAMP/checkpoint-XXXX",
        "test": "./preprocessed_data/pashto_finetuned_teacher_labels/test.csv",
        "teacher_results": "./teachers/pashto_teacher_TIMESTAMP/test_results.json",
        "student_results": "./students/pashto_student_fast_TIMESTAMP/test_results.json",
    },
    "hausa": {
        "model": "./students/hausa_student_fast_TIMESTAMP/checkpoint-XXXX",
        "test": "./preprocessed_data/hausa_finetuned_teacher_labels/test.csv",
        "teacher_results": "./teachers/hausa_teacher_TIMESTAMP/test_results.json",
        "student_results": "./students/hausa_student_fast_TIMESTAMP/test_results.json",
    },
    "burmese": {
        "model": "./students/burmese_student_fast_TIMESTAMP/checkpoint-XXXX",
        "test": "./preprocessed_data/burmese_finetuned_teacher_labels/test.csv",
        "teacher_results": "./teachers/burmese_teacher_TIMESTAMP/test_results.json",
        "student_results": "./students/burmese_student_fast_TIMESTAMP/test_results.json",
    },
}

BATCH_SIZE = 16
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256

def generate_predictions(model_path, test_csv, batch_size=16):
    """Generate predictions from a student model on test set."""
    print(f"  Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda").eval()

    df = pd.read_csv(test_csv)
    predictions = []
    references = []

    for i in tqdm(range(0, len(df), batch_size), desc="  Generating"):
        batch_texts = df["text"].iloc[i:i+batch_size].tolist()
        batch_refs = df["summary"].iloc[i:i+batch_size].tolist()

        inputs = tokenizer(
            [str(t) for t in batch_texts],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_TARGET_LENGTH,
                num_beams=4,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)
        references.extend([str(r) for r in batch_refs])

    del model
    torch.cuda.empty_cache()
    return predictions, references


def compute_bleu(predictions, references):
    """Compute BLEU score using sacrebleu."""
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return round(bleu.score / 100, 4)  # Normalize to 0-1


def compute_bertscore(predictions, references, lang):
    """Compute BERTScore."""
    from bert_score import score as bert_score

    # Map language to bert_score lang code
    lang_map = {
        "hindi": "hi", "urdu": "ur", "russian": "ru",
        "portuguese": "pt", "persian": "fa",
    }
    lang_code = lang_map.get(lang, "en")

    P, R, F1 = bert_score(predictions, references, lang=lang_code, verbose=False)
    return {
        "precision": round(P.mean().item(), 4),
        "recall": round(R.mean().item(), 4),
        "f1": round(F1.mean().item(), 4),
    }


def compute_semantic_similarity(predictions, references):
    """Compute semantic similarity using sentence-transformers."""
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Encode in batches
    pred_embs = model.encode(predictions, batch_size=64, show_progress_bar=False)
    ref_embs = model.encode(references, batch_size=64, show_progress_bar=False)

    # Pairwise cosine similarity
    sims = [
        util.cos_sim(pred_embs[i], ref_embs[i]).item()
        for i in range(len(predictions))
    ]
    
    del model
    torch.cuda.empty_cache()
    return round(np.mean(sims), 4)


def main():
    all_results = {}

    for lang, cfg in STUDENTS.items():
        print(f"\n{'='*80}")
        print(f"EVALUATING: {lang.upper()}")
        print(f"{'='*80}")

        # Load existing ROUGE scores
        with open(cfg["teacher_results"]) as f:
            teacher = json.load(f)
        with open(cfg["student_results"]) as f:
            student = json.load(f)

        # Generate predictions for new metrics
        predictions, references = generate_predictions(cfg["model"], cfg["test"])

        # Compute new metrics
        print(f"  Computing BLEU...")
        bleu = compute_bleu(predictions, references)

        print(f"  Computing BERTScore...")
        bertscore = compute_bertscore(predictions, references, lang)

        print(f"  Computing Semantic Similarity...")
        sem_sim = compute_semantic_similarity(predictions, references)

        # Retention
        teacher_rougeL = teacher["test_rougeL"]
        student_rougeL = student["test_rougeL"]
        retention = round((student_rougeL / teacher_rougeL) * 100, 2) if teacher_rougeL > 0 else 0

        all_results[lang] = {
            "teacher": {
                "rouge1": round(teacher["test_rouge1"], 4),
                "rouge2": round(teacher["test_rouge2"], 4),
                "rougeL": round(teacher["test_rougeL"], 4),
            },
            "student": {
                "rouge1": round(student["test_rouge1"], 4),
                "rouge2": round(student["test_rouge2"], 4),
                "rougeL": round(student["test_rougeL"], 4),
                "bleu": bleu,
                "bertscore": bertscore,
                "semantic_similarity": sem_sim,
            },
            "retention": {
                "rougeL_retention_pct": retention,
            },
        }

        print(f"\n  Teacher ROUGE-L: {teacher_rougeL:.4f}")
        print(f"  Student ROUGE-L: {student_rougeL:.4f}")
        print(f"  Retention: {retention}%")
        print(f"  BLEU: {bleu}")
        print(f"  BERTScore F1: {bertscore['f1']}")
        print(f"  Semantic Similarity: {sem_sim}")

    # Save
    with open("training_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n\n{'='*80}")
    print("ALL RESULTS SAVED TO training_results.json")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
