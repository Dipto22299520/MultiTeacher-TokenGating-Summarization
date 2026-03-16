"""
Preprocess XLSum dataset for multilingual knowledge distillation.
Filter for specific South Asian languages and articles with <800 tokens.
"""

import pandas as pd
import os
from transformers import AutoTokenizer
from tqdm import tqdm

# Configuration
INPUT_FILE = "xlsum_all_train.csv"
OUTPUT_DIR = "./preprocessed_data"

# Target languages (South Asian)
TARGET_LANGUAGES = ["hindi", "nepali", "sinhala", "urdu", "gujarati"]

# Token limit
MAX_TOKENS = 512

# Train/val/test split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Tokenizer (use mT5 since that's our model)
TOKENIZER_NAME = "google/mt5-small"

def count_tokens(text, tokenizer):
    """Count the number of tokens in a text."""
    try:
        tokens = tokenizer.encode(str(text), add_special_tokens=True)
        return len(tokens)
    except:
        return 0

def process_language(df, language, tokenizer):
    """Process data for a specific language."""
    print(f"\n{'='*80}")
    print(f"Processing: {language.upper()}")
    print(f"{'='*80}")
    
    # Filter for the language
    lang_df = df[df['language'].str.lower() == language.lower()].copy()
    print(f"Total {language} samples: {len(lang_df)}")
    
    if len(lang_df) == 0:
        print(f"WARNING: No data found for {language}")
        return
    
    # Count tokens for each article
    print(f"Counting tokens (this may take a while)...")
    tqdm.pandas(desc=f"Tokenizing {language}")
    lang_df['token_count'] = lang_df['text'].progress_apply(lambda x: count_tokens(x, tokenizer))
    
    # Filter by token count
    filtered_df = lang_df[lang_df['token_count'] < MAX_TOKENS].copy()
    print(f"Samples with <{MAX_TOKENS} tokens: {len(filtered_df)} ({len(filtered_df)/len(lang_df)*100:.1f}%)")
    
    if len(filtered_df) == 0:
        print(f"WARNING: No samples under {MAX_TOKENS} tokens for {language}")
        return
    
    # Shuffle the data
    filtered_df = filtered_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train/val/test
    n = len(filtered_df)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    
    train_df = filtered_df[:train_end]
    val_df = filtered_df[train_end:val_end]
    test_df = filtered_df[val_end:]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    # Create language-specific directory
    lang_dir = os.path.join(OUTPUT_DIR, language)
    os.makedirs(lang_dir, exist_ok=True)
    
    # Save only the required columns: text and summary
    columns_to_save = ['text', 'summary']
    
    train_path = os.path.join(lang_dir, "train.csv")
    val_path = os.path.join(lang_dir, "val.csv")
    test_path = os.path.join(lang_dir, "test.csv")
    
    train_df[columns_to_save].to_csv(train_path, index=False)
    val_df[columns_to_save].to_csv(val_path, index=False)
    test_df[columns_to_save].to_csv(test_path, index=False)
    
    print(f"\nSaved to:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    
    # Save statistics
    stats = {
        'language': language,
        'total_samples': len(lang_df),
        'filtered_samples': len(filtered_df),
        'filter_percentage': len(filtered_df)/len(lang_df)*100,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'avg_tokens': filtered_df['token_count'].mean(),
        'median_tokens': filtered_df['token_count'].median(),
        'max_tokens': filtered_df['token_count'].max(),
    }
    
    stats_path = os.path.join(lang_dir, "stats.txt")
    with open(stats_path, 'w', encoding='utf-8') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    return stats

def main():
    print(f"\n{'='*80}")
    print("MULTILINGUAL XLSUM PREPROCESSING")
    print(f"{'='*80}")
    print(f"\nInput file: {INPUT_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target languages: {', '.join(TARGET_LANGUAGES)}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Split: {TRAIN_RATIO*100:.0f}% train, {VAL_RATIO*100:.0f}% val, {TEST_RATIO*100:.0f}% test")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    
    # Read the CSV file in chunks to handle large file
    print(f"\nReading {INPUT_FILE}...")
    print("(This may take a while for large files)")
    
    # First, let's check available languages
    print("\nScanning for available languages...")
    available_langs = set()
    for chunk in pd.read_csv(INPUT_FILE, usecols=['language'], chunksize=100000):
        available_langs.update(chunk['language'].str.lower().unique())
    
    print(f"\nAvailable languages in dataset: {sorted(available_langs)}")
    
    # Check which target languages are available
    missing_langs = [lang for lang in TARGET_LANGUAGES if lang.lower() not in available_langs]
    if missing_langs:
        print(f"\nWARNING: These languages not found in dataset: {missing_langs}")
    
    present_langs = [lang for lang in TARGET_LANGUAGES if lang.lower() in available_langs]
    print(f"\nWill process these languages: {present_langs}")
    
    # Load the full dataset (only target languages)
    print(f"\nLoading data for target languages...")
    df_list = []
    for chunk in pd.read_csv(INPUT_FILE, chunksize=100000):
        chunk_filtered = chunk[chunk['language'].str.lower().isin([l.lower() for l in TARGET_LANGUAGES])]
        if len(chunk_filtered) > 0:
            df_list.append(chunk_filtered)
    
    df = pd.concat(df_list, ignore_index=True)
    print(f"Total samples loaded: {len(df)}")
    
    # Process each language
    all_stats = []
    for language in present_langs:
        stats = process_language(df, language, tokenizer)
        if stats:
            all_stats.append(stats)
    
    # Save overall summary
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("MULTILINGUAL PREPROCESSING SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Max tokens filter: {MAX_TOKENS}\n")
        f.write(f"Tokenizer: {TOKENIZER_NAME}\n\n")
        
        for stats in all_stats:
            f.write(f"\n{stats['language'].upper()}:\n")
            f.write(f"  Total samples: {stats['total_samples']}\n")
            f.write(f"  Filtered samples (<{MAX_TOKENS} tokens): {stats['filtered_samples']} ({stats['filter_percentage']:.1f}%)\n")
            f.write(f"  Train/Val/Test: {stats['train_samples']}/{stats['val_samples']}/{stats['test_samples']}\n")
            f.write(f"  Token stats: avg={stats['avg_tokens']:.1f}, median={stats['median_tokens']:.1f}, max={stats['max_tokens']}\n")
    
    print(f"\nSummary saved to: {summary_path}")
    print(f"\nAll datasets saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Review the statistics for each language")
    print("  2. Update train_teacher.py and train_student.py to point to the language-specific directories")
    print("  3. Run teacher training for each language")
    print("  4. Run student distillation for each language")

if __name__ == "__main__":
    main()
