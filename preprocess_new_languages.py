"""
Preprocess XLSum dataset for 5 new low-resource languages:
  - Nepali   (Devanagari script, South Asia)
  - Amharic  (Ge'ez/Ethiopic script, East Africa)
  - Pashto   (Arabic-variant script, Afghanistan/Pakistan)
  - Hausa    (Latin script, West Africa)
  - Burmese  (Myanmar script, Southeast Asia)

Filters articles to <= 512 tokens and creates train/val/test splits.
Run this BEFORE train_new_languages.py.
"""

import pandas as pd
import os
from transformers import AutoTokenizer
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

INPUT_FILE = "xlsum_all_train.csv"
OUTPUT_DIR = "./preprocessed_data"

# The 5 new low-resource languages (must match the 'language' column in xlsum_all_train.csv)
TARGET_LANGUAGES = ["nepali", "amharic", "pashto", "hausa", "burmese"]

MAX_TOKENS = 512
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# If a language has fewer than this many filtered samples, use ALL of them for
# training (maximise data for low-resource languages).  A small held-out slice
# is still carved out for val/test so early-stopping and ROUGE evaluation work,
# but that slice also appears in train.csv so no article is wasted.
SMALL_DATASET_THRESHOLD = 8000
SMALL_VAL_TEST_SIZE = 200   # samples reserved for val/test in small-dataset mode

TOKENIZER_NAME = "google/mt5-small"

# ============================================================================
# Helpers
# ============================================================================

def count_tokens(text, tokenizer):
    try:
        return len(tokenizer.encode(str(text), add_special_tokens=True))
    except Exception:
        return 0


def process_language(df, language, tokenizer):
    print(f"\n{'=' * 80}")
    print(f"Processing: {language.upper()}")
    print(f"{'=' * 80}")

    lang_df = df[df["language"].str.lower() == language.lower()].copy()
    total = len(lang_df)
    print(f"Total {language} samples: {total}")

    if total == 0:
        print(f"WARNING: No data found for '{language}' in {INPUT_FILE}.")
        print(f"  Check that the language column contains this exact name.")
        return

    # Token-count filter
    tqdm.pandas(desc=f"Tokenising {language}")
    lang_df["token_count"] = lang_df["text"].progress_apply(
        lambda x: count_tokens(x, tokenizer)
    )

    filtered_df = lang_df[lang_df["token_count"] <= MAX_TOKENS].copy()
    kept_pct = len(filtered_df) / total * 100
    print(f"Samples with <= {MAX_TOKENS} tokens: {len(filtered_df)} ({kept_pct:.1f}%)")

    if len(filtered_df) == 0:
        print(f"WARNING: No samples under {MAX_TOKENS} tokens for {language}. Skipping.")
        return

    # Shuffle
    filtered_df = filtered_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(filtered_df)

    if n < SMALL_DATASET_THRESHOLD:
        # Low-resource mode: train on ALL data so nothing is wasted.
        # A small tail slice is used for val/test (monitoring + ROUGE eval);
        # it intentionally overlaps with train — acceptable for low-resource.
        held = min(SMALL_VAL_TEST_SIZE, max(50, int(n * 0.05)))
        train_df = filtered_df[["text", "summary"]]          # all samples
        val_df   = filtered_df.tail(held)[["text", "summary"]]
        test_df  = filtered_df.tail(held)[["text", "summary"]]
        print(f"\n[LOW-RESOURCE MODE] < {SMALL_DATASET_THRESHOLD} samples — using ALL {n} for training.")
        print(f"  Val/Test : {held} samples (tail slice, overlaps with train)")
    else:
        # Normal 80/10/10 split
        train_end = int(n * TRAIN_RATIO)
        val_end   = train_end + int(n * VAL_RATIO)
        train_df = filtered_df[:train_end][["text", "summary"]]
        val_df   = filtered_df[train_end:val_end][["text", "summary"]]
        test_df  = filtered_df[val_end:][["text", "summary"]]
        print(f"\n[NORMAL MODE] >= {SMALL_DATASET_THRESHOLD} samples — 80/10/10 split.")

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")

    # Save
    lang_dir = os.path.join(OUTPUT_DIR, language)
    os.makedirs(lang_dir, exist_ok=True)

    train_df.to_csv(os.path.join(lang_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(lang_dir, "val.csv"),   index=False)
    test_df.to_csv(os.path.join(lang_dir, "test.csv"),  index=False)

    print(f"Saved to: {lang_dir}/")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("PREPROCESSING NEW LOW-RESOURCE LANGUAGES")
    print(f"Languages: {', '.join(l.capitalize() for l in TARGET_LANGUAGES)}")
    print("=" * 80)

    print(f"\nLoading tokeniser ({TOKENIZER_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=False)

    print(f"\nLoading dataset from {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Total rows: {len(df)}")

    # Show all unique language names to help debug mismatches
    if "language" in df.columns:
        all_langs = sorted(df["language"].str.lower().unique())
        print(f"\nAll languages in CSV ({len(all_langs)} total):")
        print("  " + ", ".join(all_langs))
    else:
        print("WARNING: No 'language' column found in CSV. Check column names:", list(df.columns))
        return

    for lang in TARGET_LANGUAGES:
        process_language(df, lang, tokenizer)

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("Next step: python train_new_languages.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
