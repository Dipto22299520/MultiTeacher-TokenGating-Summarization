"""
Count <=512 token samples for Russian, Portuguese, Persian, Gujarati
Then create preprocessed datasets for all of them
"""

import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import os

tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
langs = ["gujarati", "russian", "portuguese", "persian"]

MAX_TOKENS = 512
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
OUTPUT_DIR = "./preprocessed_data"

# Low-resource threshold: if filtered samples < this, use ALL for training
SMALL_DATASET_THRESHOLD = 8000
SMALL_VAL_TEST_SIZE = 200

print("Loading full CSV...")
df = pd.read_csv("xlsum_all_train.csv")
print(f"Total rows: {len(df)}")

for lang in langs:
    print(f"\n{'='*80}")
    print(f"Processing: {lang.upper()}")
    print(f"{'='*80}")
    
    lang_df = df[df["language"].str.lower() == lang].copy()
    total = len(lang_df)
    print(f"Total {lang} samples: {total}")
    
    if total == 0:
        print(f"No data found for {lang}, skipping.")
        continue
    
    # Count tokens
    tqdm.pandas(desc=f"Tokenizing {lang}")
    lang_df["token_count"] = lang_df["text"].progress_apply(
        lambda x: len(tokenizer.encode(str(x), add_special_tokens=True))
    )
    
    filtered_df = lang_df[lang_df["token_count"] <= MAX_TOKENS].copy()
    print(f"Samples with <={MAX_TOKENS} tokens: {len(filtered_df)} ({len(filtered_df)/total*100:.1f}%)")
    
    # Shuffle and split
    filtered_df = filtered_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(filtered_df)

    if n < SMALL_DATASET_THRESHOLD:
        held = min(SMALL_VAL_TEST_SIZE, max(50, int(n * 0.05)))
        train_df = filtered_df[["text", "summary"]]
        val_df   = filtered_df.tail(held)[["text", "summary"]]
        test_df  = filtered_df.tail(held)[["text", "summary"]]
        print(f"  [LOW-RESOURCE] < {SMALL_DATASET_THRESHOLD} samples — using ALL {n} for training.")
        print(f"  Val/Test: {held} samples (tail slice, overlaps train)")
    else:
        train_end = int(n * TRAIN_RATIO)
        val_end   = train_end + int(n * VAL_RATIO)
        train_df = filtered_df[:train_end][["text", "summary"]]
        val_df   = filtered_df[train_end:val_end][["text", "summary"]]
        test_df  = filtered_df[val_end:][["text", "summary"]]
    
    # Save
    lang_dir = os.path.join(OUTPUT_DIR, lang)
    os.makedirs(lang_dir, exist_ok=True)
    train_df.to_csv(os.path.join(lang_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(lang_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(lang_dir, "test.csv"), index=False)
    
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")
    print(f"  Saved to: {lang_dir}/")

print("\nDone!")
