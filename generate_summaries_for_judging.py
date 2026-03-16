"""
Generate summaries from A2 and A5 best models for multi-judge evaluation.
==========================================================================

Each model is evaluated on its OWN test data:
  - Text Summarization models (A2, A5)       → data/test.csv (Hasan et al.)
  - BanSum models (A2_bansum, A5_bansum)     → BanSum test split (80/10/10, seed=42)

Randomly selects 1000 test articles per model, generates summaries, and
saves them as CSV + metadata JSON for LLM-based multi-judge evaluation.

Output:
  judge_evaluation_samples.csv   — 4000 rows (1000 per model × 4 models)
  judge_evaluation_metadata.json — generation config & timing stats

Usage:
    py -3.12 generate_summaries_for_judging.py
"""

import os
import json
import random
import time
import pandas as pd
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

SEED = 42
NUM_SAMPLES = 1000
MAX_TARGET_LENGTH = 256
BATCH_SIZE = 32  # generation batch size
NUM_BEAMS = 6

TEXT_SUM_TEST_FILE = "data/test.csv"  # Hasan et al. text_summarization dataset
BANSUM_FILE = "bansum_lte_1000_tokens.json"
OUTPUT_CSV = "judge_evaluation_samples.csv"
OUTPUT_META = "judge_evaluation_metadata.json"

# Each entry: (model_name, model_path, dataset, max_input_length)
MODELS = [
    ("A2_single_kd",            "ablation_results/A2_single_kd_20260222_093225/best_model",                        "text_summarization",  512),
    ("A5_full_pipeline",         "ablation_results/A5_full_pipeline_20260223_005736/best_model",                     "text_summarization",  512),
    ("A2_single_kd_bansum",      "ablation_results_bansum/A2_single_kd_20260225_120221/best_model",                  "bansum", 850),
    ("A5_full_pipeline_bansum",  "ablation_results_bansum/A5_full_pipeline_20260228_034939_batch16_backup/best_model", "bansum", 850),
]

# ============================================================================
# Main
# ============================================================================

def load_text_summarization_test():
    """Load Text Summarization (Hasan et al.) test data from CSV."""
    print(f"\nLoading Text Summarization test data from {TEXT_SUM_TEST_FILE}...")
    df = pd.read_csv(TEXT_SUM_TEST_FILE)
    print(f"  Total Text Summarization test samples: {len(df)}")
    return df


def load_bansum_test():
    """Load BanSum data and extract test split (same 80/10/10 split as training)."""
    print(f"\nLoading BanSum data from {BANSUM_FILE}...")
    with open(BANSUM_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame([{"text": item["main"], "summary": item["sum2"]} for item in data])
    # Same deterministic split as train_teacher_bansum.py / train_student_ablation_bansum.py
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))
    test_df = df[train_size + val_size:].reset_index(drop=True)
    print(f"  Total BanSum samples: {len(df)}, Test split: {len(test_df)}")
    return test_df


def select_samples(test_df, dataset_name):
    """Randomly select NUM_SAMPLES from a test dataframe."""
    if NUM_SAMPLES > len(test_df):
        raise ValueError(f"Requested {NUM_SAMPLES} samples but {dataset_name} test has only {len(test_df)}")
    selected_indices = sorted(random.sample(range(len(test_df)), NUM_SAMPLES))
    selected_df = test_df.iloc[selected_indices].reset_index(drop=True)
    articles = selected_df["text"].tolist()
    references = selected_df["summary"].tolist()
    categories = selected_df["category"].tolist() if "category" in selected_df.columns else [None] * len(selected_df)
    return articles, references, categories, selected_indices


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Pre-load both test sets
    textsum_test_df = load_text_summarization_test()
    bansum_test_df = load_bansum_test()

    # Select samples per dataset (separate RNG draws to keep seeds independent)
    rng_textsum = random.Random(SEED)
    rng_bansum = random.Random(SEED)
    textsum_indices = sorted(rng_textsum.sample(range(len(textsum_test_df)), NUM_SAMPLES))
    bansum_indices = sorted(rng_bansum.sample(range(len(bansum_test_df)), NUM_SAMPLES))

    dataset_cache = {
        "text_summarization": {
            "test_df": textsum_test_df,
            "indices": textsum_indices,
            "articles": textsum_test_df.iloc[textsum_indices]["text"].tolist(),
            "references": textsum_test_df.iloc[textsum_indices]["summary"].tolist(),
            "categories": textsum_test_df.iloc[textsum_indices]["category"].tolist() if "category" in textsum_test_df.columns else [None] * NUM_SAMPLES,
            "source_file": TEXT_SUM_TEST_FILE,
            "total_test": len(textsum_test_df),
        },
        "bansum": {
            "test_df": bansum_test_df,
            "indices": bansum_indices,
            "articles": bansum_test_df.iloc[bansum_indices]["text"].tolist(),
            "references": bansum_test_df.iloc[bansum_indices]["summary"].tolist(),
            "categories": [None] * NUM_SAMPLES,
            "source_file": BANSUM_FILE + " (test split)",
            "total_test": len(bansum_test_df),
        },
    }

    print(f"\n  Text Summarization: selected {NUM_SAMPLES} / {len(textsum_test_df)} test samples")
    print(f"  BanSum: selected {NUM_SAMPLES} / {len(bansum_test_df)} test samples")

    all_results = []
    timing_stats = {}

    for model_name, model_path, dataset, max_input_length in MODELS:
        ds = dataset_cache[dataset]
        articles = ds["articles"]
        references = ds["references"]
        categories = ds["categories"]
        original_indices = ds["indices"]

        print(f"\n{'='*70}")
        print(f"Generating summaries with: {model_name}")
        print(f"  Model path: {model_path}")
        print(f"  Dataset: {dataset} | MAX_INPUT: {max_input_length} | Samples: {len(articles)}")
        print(f"{'='*70}")

        if not os.path.isdir(model_path):
            print(f"  ERROR: Model path not found: {model_path}")
            continue

        # Load model and tokenizer
        load_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.to(device)
        model.eval()
        load_time = time.time() - load_start

        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,} ({params/1e6:.1f}M)")
        print(f"  Model load time: {load_time:.2f}s")

        # Generate in batches with timing
        generated_summaries = []
        per_sample_times = []
        num_batches = (len(articles) + BATCH_SIZE - 1) // BATCH_SIZE

        gen_start = time.time()
        with torch.no_grad():
            for i in tqdm(range(0, len(articles), BATCH_SIZE),
                         total=num_batches, desc=f"  {model_name}"):
                batch_texts = articles[i : i + BATCH_SIZE]
                batch_size_actual = len(batch_texts)

                batch_start = time.time()
                inputs = tokenizer(
                    batch_texts,
                    max_length=max_input_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                outputs = model.generate(
                    **inputs,
                    max_length=MAX_TARGET_LENGTH,
                    num_beams=NUM_BEAMS,
                    length_penalty=1.0,
                    early_stopping=True,
                )
                batch_elapsed = time.time() - batch_start
                per_sample_time = batch_elapsed / batch_size_actual

                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                generated_summaries.extend(decoded)
                per_sample_times.extend([per_sample_time] * batch_size_actual)

        total_gen_time = time.time() - gen_start
        avg_time_per_sample = total_gen_time / len(generated_summaries)
        throughput = len(generated_summaries) / total_gen_time

        print(f"  Generated {len(generated_summaries)} summaries")
        print(f"  Total generation time: {total_gen_time:.1f}s")
        print(f"  Avg time/sample: {avg_time_per_sample:.3f}s")
        print(f"  Throughput: {throughput:.1f} samples/s")

        timing_stats[model_name] = {
            "dataset": dataset,
            "max_input_length": max_input_length,
            "test_source": ds["source_file"],
            "total_test_samples": ds["total_test"],
            "model_load_time_sec": round(load_time, 3),
            "total_generation_time_sec": round(total_gen_time, 3),
            "avg_time_per_sample_sec": round(avg_time_per_sample, 4),
            "throughput_samples_per_sec": round(throughput, 2),
            "num_parameters": params,
            "num_parameters_M": round(params / 1e6, 1),
        }

        # Build result entries
        for idx, (article, reference, generated, category, orig_idx, sample_time) in enumerate(
            zip(articles, references, generated_summaries, categories, original_indices, per_sample_times)
        ):
            all_results.append({
                "sample_id": idx,
                "original_test_index": orig_idx,
                "dataset": dataset,
                "category": category,
                "article": article,
                "reference_summary": reference,
                "model_summary": generated,
                "model_name": model_name,
                "generation_time_sec": round(sample_time, 4),
            })

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Save CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # Save metadata + timing as JSON sidecar
    metadata = {
        "description": "Summaries from A2 and A5 distilled models for multi-judge LLM evaluation. "
                       "Each model evaluated on its own training domain test data.",
        "num_samples_per_model": NUM_SAMPLES,
        "num_entries": len(all_results),
        "models": [m[0] for m in MODELS],
        "seed": SEED,
        "generation_config": {
            "num_beams": NUM_BEAMS,
            "max_target_length": MAX_TARGET_LENGTH,
            "batch_size": BATCH_SIZE,
        },
        "test_sources": {
            "text_summarization": {"file": TEXT_SUM_TEST_FILE, "source": "Hasan et al. text_summarization.csv", "total_test": len(textsum_test_df), "selected": NUM_SAMPLES},
            "bansum": {"file": BANSUM_FILE, "split": "80/10/10 seed=42 test", "total_test": len(bansum_test_df), "selected": NUM_SAMPLES},
        },
        "device": str(device),
        "timing": timing_stats,
        "generated_at": datetime.now().isoformat(),
    }

    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"DONE")
    print(f"  CSV:      {OUTPUT_CSV}  ({len(all_results)} rows)")
    print(f"  Metadata: {OUTPUT_META}")
    print(f"  {NUM_SAMPLES} articles x {len(MODELS)} models = {len(all_results)} total")
    print(f"  Text Summarization models used: {TEXT_SUM_TEST_FILE} ({len(textsum_test_df)} test)")
    print(f"  BanSum models used: {BANSUM_FILE} test split ({len(bansum_test_df)} test)")
    print(f"\n  Inference timing (distillation comparison):")
    for mname, stats in timing_stats.items():
        print(f"    {mname}: {stats['avg_time_per_sample_sec']}s/sample, "
              f"{stats['throughput_samples_per_sec']} samples/s, "
              f"{stats['num_parameters_M']}M params")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
