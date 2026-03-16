"""
Comprehensive evaluation of all student models.
Computes ROUGE-1/2/L, BLEU, BERTScore, and Semantic Similarity.
Generates final results JSON.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score_fn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

LANGUAGES = {
    "hindi": {
        "teacher_dir": "./teachers/hindi_teacher_20260306_181347",
        "student_dir": "./students/hindi_student_fast_20260306_212437",
        "test_file": "./preprocessed_data/hindi/test.csv",
    },
    "urdu": {
        "teacher_dir": "./teachers/urdu_teacher_20260306_213245",
        "student_dir": "./students/urdu_student_fast_20260306_230834",
        "test_file": "./preprocessed_data/urdu/test.csv",
    },
    "russian": {
        "teacher_dir": "./teachers/russian_teacher_20260307_000405",
        "student_dir": "./students/russian_student_fast_20260307_013515",
        "test_file": "./preprocessed_data/russian/test.csv",
    },
    "portuguese": {
        "teacher_dir": "./teachers/portuguese_teacher_20260307_014259",
        "student_dir": "./students/portuguese_student_fast_20260307_031722",
        "test_file": "./preprocessed_data/portuguese/test.csv",
    },
    "persian": {
        "teacher_dir": "./teachers/persian_teacher_20260307_032340",
        "student_dir": "./students/persian_student_fast_20260307_041138",
        "test_file": "./preprocessed_data/persian/test.csv",
    },
    # -------------------------------------------------------------------------
    # New low-resource languages — fill in the timestamped paths after training
    # -------------------------------------------------------------------------
    "nepali": {
        "teacher_dir": "./teachers/nepali_teacher_TIMESTAMP",       # update after training
        "student_dir": "./students/nepali_student_fast_TIMESTAMP",  # update after training
        "test_file": "./preprocessed_data/nepali/test.csv",
    },
    "amharic": {
        "teacher_dir": "./teachers/amharic_teacher_TIMESTAMP",
        "student_dir": "./students/amharic_student_fast_TIMESTAMP",
        "test_file": "./preprocessed_data/amharic/test.csv",
    },
    "pashto": {
        "teacher_dir": "./teachers/pashto_teacher_TIMESTAMP",
        "student_dir": "./students/pashto_student_fast_TIMESTAMP",
        "test_file": "./preprocessed_data/pashto/test.csv",
    },
    "hausa": {
        "teacher_dir": "./teachers/hausa_teacher_TIMESTAMP",
        "student_dir": "./students/hausa_student_fast_TIMESTAMP",
        "test_file": "./preprocessed_data/hausa/test.csv",
    },
    "burmese": {
        "teacher_dir": "./teachers/burmese_teacher_TIMESTAMP",
        "student_dir": "./students/burmese_student_fast_TIMESTAMP",
        "test_file": "./preprocessed_data/burmese/test.csv",
    },
}

MAX_EVAL_SAMPLES = 200  # Evaluate on up to 200 test samples per language
BATCH_SIZE = 16

def generate_predictions(model_path, test_df, max_samples=200):
    """Generate predictions from a model."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda").eval()
    
    df = test_df.head(max_samples)
    predictions = []
    
    for i in range(0, len(df), BATCH_SIZE):
        batch_texts = ["summarize: " + str(t) for t in df["text"].iloc[i:i+BATCH_SIZE].tolist()]
        inputs = tokenizer(
            batch_texts, max_length=512, truncation=True, padding=True, return_tensors="pt"
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=256, num_beams=6, early_stopping=True)
        
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(preds)
    
    del model
    torch.cuda.empty_cache()
    return predictions


def compute_rouge(predictions, references):
    """Compute ROUGE-1, ROUGE-2, ROUGE-L."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    
    for pred, ref in zip(predictions, references):
        s = scorer.score(str(ref), str(pred))
        scores["rouge1"].append(s["rouge1"].fmeasure)
        scores["rouge2"].append(s["rouge2"].fmeasure)
        scores["rougeL"].append(s["rougeL"].fmeasure)
    
    return {k: round(float(np.mean(v)), 4) for k, v in scores.items()}


def compute_bleu(predictions, references):
    """Compute BLEU score."""
    bleu = BLEU()
    # sacrebleu expects list of strings for hypotheses, list of list of strings for references
    result = bleu.corpus_score(
        [str(p) for p in predictions],
        [[str(r) for r in references]]
    )
    return round(result.score, 4)


def compute_bertscore(predictions, references, lang="en"):
    """Compute BERTScore using multilingual model."""
    P, R, F1 = bert_score_fn(
        [str(p) for p in predictions],
        [str(r) for r in references],
        model_type="bert-base-multilingual-cased",
        num_layers=9,
        batch_size=32,
        verbose=False
    )
    return {
        "precision": round(float(P.mean()), 4),
        "recall": round(float(R.mean()), 4),
        "f1": round(float(F1.mean()), 4),
    }


def compute_semantic_similarity(predictions, references):
    """Compute semantic similarity using sentence-transformers."""
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    pred_embeddings = model.encode([str(p) for p in predictions], batch_size=32, show_progress_bar=False)
    ref_embeddings = model.encode([str(r) for r in references], batch_size=32, show_progress_bar=False)
    
    # Pairwise cosine similarity (diagonal)
    similarities = []
    for i in range(len(predictions)):
        sim = cosine_similarity([pred_embeddings[i]], [ref_embeddings[i]])[0][0]
        similarities.append(float(sim))
    
    del model
    torch.cuda.empty_cache()
    return round(float(np.mean(similarities)), 4)


def evaluate_language(language, config):
    """Run full evaluation for one language."""
    print(f"\n{'#'*80}")
    print(f"# EVALUATING: {language.upper()}")
    print(f"{'#'*80}")
    
    test_df = pd.read_csv(config["test_file"]).head(MAX_EVAL_SAMPLES)
    references = test_df["summary"].tolist()
    
    results = {"language": language, "num_test_samples": len(test_df)}
    
    # --- Teacher metrics ---
    teacher_path = config["teacher_dir"]
    teacher_test_results = os.path.join(teacher_path, "test_results.json")
    
    if os.path.exists(teacher_test_results):
        with open(teacher_test_results) as f:
            tr = json.load(f)
        results["teacher"] = {
            "rouge1": round(tr.get("test_rouge1", 0), 4),
            "rouge2": round(tr.get("test_rouge2", 0), 4),
            "rougeL": round(tr.get("test_rougeL", 0), 4),
        }
        print(f"Teacher ROUGE (from saved): R1={results['teacher']['rouge1']}, R2={results['teacher']['rouge2']}, RL={results['teacher']['rougeL']}")
    
    # --- Student evaluation ---
    student_path = config["student_dir"]
    print(f"\nGenerating student predictions from: {student_path}")
    student_preds = generate_predictions(student_path, test_df, MAX_EVAL_SAMPLES)
    
    # ROUGE
    print("Computing ROUGE...")
    student_rouge = compute_rouge(student_preds, references)
    
    # BLEU
    print("Computing BLEU...")
    student_bleu = compute_bleu(student_preds, references)
    
    # BERTScore
    print("Computing BERTScore...")
    student_bertscore = compute_bertscore(student_preds, references)
    
    # Semantic Similarity
    print("Computing Semantic Similarity...")
    student_semsim = compute_semantic_similarity(student_preds, references)
    
    results["student"] = {
        "rouge1": student_rouge["rouge1"],
        "rouge2": student_rouge["rouge2"],
        "rougeL": student_rouge["rougeL"],
        "bleu": student_bleu,
        "bertscore_precision": student_bertscore["precision"],
        "bertscore_recall": student_bertscore["recall"],
        "bertscore_f1": student_bertscore["f1"],
        "semantic_similarity": student_semsim,
    }
    
    # Retention
    if "teacher" in results:
        teacher_rl = results["teacher"]["rougeL"]
        student_rl = results["student"]["rougeL"]
        results["retention_rougeL_pct"] = round((student_rl / teacher_rl) * 100, 2) if teacher_rl > 0 else 0
    
    print(f"\nStudent Results:")
    print(f"  ROUGE-1: {student_rouge['rouge1']}")
    print(f"  ROUGE-2: {student_rouge['rouge2']}")
    print(f"  ROUGE-L: {student_rouge['rougeL']}")
    print(f"  BLEU: {student_bleu}")
    print(f"  BERTScore F1: {student_bertscore['f1']}")
    print(f"  Semantic Similarity: {student_semsim}")
    if "retention_rougeL_pct" in results:
        print(f"  Retention: {results['retention_rougeL_pct']}%")
    
    return results


def main():
    print("=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)
    
    all_results = {}
    
    for language, config in LANGUAGES.items():
        if not os.path.exists(config["student_dir"]):
            print(f"\nSkipping {language}: student model not found")
            continue
        
        result = evaluate_language(language, config)
        all_results[language] = result
    
    # Save results
    output_path = "./comprehensive_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n\n{'='*80}")
    print("ALL EVALUATIONS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_path}")
    
    # Summary table
    print(f"\n{'Language':<12} {'T-RL':<8} {'S-RL':<8} {'S-BLEU':<8} {'S-BERT':<8} {'S-SemSim':<10} {'Retain%':<8}")
    print("-" * 70)
    for lang, r in all_results.items():
        t_rl = r.get("teacher", {}).get("rougeL", 0)
        s = r["student"]
        ret = r.get("retention_rougeL_pct", 0)
        print(f"{lang:<12} {t_rl:<8} {s['rougeL']:<8} {s['bleu']:<8} {s['bertscore_f1']:<8} {s['semantic_similarity']:<10} {ret:<8}")


if __name__ == "__main__":
    main()
