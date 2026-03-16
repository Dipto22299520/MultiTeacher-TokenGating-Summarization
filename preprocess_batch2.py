"""
Preprocess XLSum for: Ukrainian, Tamil, Telugu, Gujarati, Vietnamese
- No token-length filtering (use all articles)
- Standard 80/10/10 train/val/test split
"""

import pandas as pd
import os

INPUT_FILE = "xlsum_all_train.csv"
OUTPUT_DIR = "./preprocessed_data"

LANGUAGES = ["ukrainian", "tamil", "telugu", "gujarati", "vietnamese"]

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"Total rows: {len(df)}\n")

all_langs_lower = df["language"].str.lower().unique()
print(f"Available languages ({len(all_langs_lower)}): {sorted(all_langs_lower)}\n")

for lang in LANGUAGES:
    print("=" * 70)
    print(f"Processing: {lang.upper()}")
    print("=" * 70)

    lang_df = df[df["language"].str.lower() == lang].copy()
    total = len(lang_df)
    print(f"Total samples: {total}")

    if total == 0:
        print(f"WARNING: '{lang}' not found — skipping.")
        continue

    lang_df = lang_df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(lang_df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = train_end + int(n * VAL_RATIO)

    train_df = lang_df[:train_end][["text", "summary"]]
    val_df   = lang_df[train_end:val_end][["text", "summary"]]
    test_df  = lang_df[val_end:][["text", "summary"]]

    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")

    lang_dir = os.path.join(OUTPUT_DIR, lang)
    os.makedirs(lang_dir, exist_ok=True)

    train_df.to_csv(os.path.join(lang_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(lang_dir, "val.csv"),     index=False)
    test_df.to_csv(os.path.join(lang_dir, "test.csv"),   index=False)
    print(f"  Saved to: {lang_dir}/\n")

print("=" * 70)
print("DONE — next step: python train_batch2.py")
print("=" * 70)
