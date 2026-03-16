import pandas as pd
from transformers import AutoTokenizer

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")

# Load datasets
print("\nLoading datasets...")
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
test_df = pd.read_csv("data/test.csv")

print(f"Train samples: {len(train_df)}")
print(f"Val samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

# Analyze lengths
def analyze_lengths(df, name):
    print(f"\n{'='*80}")
    print(f"{name} DATASET ANALYSIS")
    print('='*80)
    
    # Tokenize texts and summaries
    text_lengths = []
    summary_lengths = []
    
    for idx, row in df.iterrows():
        text_tokens = tokenizer(str(row['text']), truncation=False)['input_ids']
        summary_tokens = tokenizer(str(row['summary']), truncation=False)['input_ids']
        
        text_lengths.append(len(text_tokens))
        summary_lengths.append(len(summary_tokens))
        
        if idx >= 2000:  # Sample first 2000 for speed
            break
    
    # Statistics
    import numpy as np
    
    print(f"\nINPUT TEXT (articles):")
    print(f"  Mean: {np.mean(text_lengths):.1f} tokens")
    print(f"  Median: {np.median(text_lengths):.1f} tokens")
    print(f"  50th percentile: {np.percentile(text_lengths, 50):.1f} tokens")
    print(f"  75th percentile: {np.percentile(text_lengths, 75):.1f} tokens")
    print(f"  90th percentile: {np.percentile(text_lengths, 90):.1f} tokens")
    print(f"  95th percentile: {np.percentile(text_lengths, 95):.1f} tokens")
    print(f"  99th percentile: {np.percentile(text_lengths, 99):.1f} tokens")
    print(f"  Max: {np.max(text_lengths)} tokens")
    
    print(f"\nTARGET SUMMARIES:")
    print(f"  Mean: {np.mean(summary_lengths):.1f} tokens")
    print(f"  Median: {np.median(summary_lengths):.1f} tokens")
    print(f"  50th percentile: {np.percentile(summary_lengths, 50):.1f} tokens")
    print(f"  75th percentile: {np.percentile(summary_lengths, 75):.1f} tokens")
    print(f"  90th percentile: {np.percentile(summary_lengths, 90):.1f} tokens")
    print(f"  95th percentile: {np.percentile(summary_lengths, 95):.1f} tokens")
    print(f"  99th percentile: {np.percentile(summary_lengths, 99):.1f} tokens")
    print(f"  Max: {np.max(summary_lengths)} tokens")
    
    # Truncation analysis
    print(f"\n📊 TRUNCATION IMPACT:")
    print(f"  With MAX_INPUT=384: {np.sum(np.array(text_lengths) > 384)/len(text_lengths)*100:.1f}% texts truncated")
    print(f"  With MAX_INPUT=512: {np.sum(np.array(text_lengths) > 512)/len(text_lengths)*100:.1f}% texts truncated")
    print(f"  With MAX_TARGET=128: {np.sum(np.array(summary_lengths) > 128)/len(summary_lengths)*100:.1f}% summaries truncated")
    print(f"  With MAX_TARGET=256: {np.sum(np.array(summary_lengths) > 256)/len(summary_lengths)*100:.1f}% summaries truncated")
    
    return text_lengths, summary_lengths

# Analyze train set
train_text_lens, train_sum_lens = analyze_lengths(train_df, "TRAIN")

# Analyze validation set
val_text_lens, val_sum_lens = analyze_lengths(val_df, "VALIDATION")

print(f"\n{'='*80}")
print("RECOMMENDATIONS")
print('='*80)

import numpy as np

# Combine all lengths for overall stats
all_text_lens = train_text_lens + val_text_lens
all_sum_lens = train_sum_lens + val_sum_lens

text_95 = np.percentile(all_text_lens, 95)
sum_95 = np.percentile(all_sum_lens, 95)

print(f"\n✓ RECOMMENDED SETTINGS (covers 95% of data):")
print(f"  MAX_INPUT_LENGTH = {int(text_95)}")
print(f"  MAX_TARGET_LENGTH = {int(sum_95)}")

print(f"\n⚡ SPEED-OPTIMIZED SETTINGS (covers ~85-90%):")
print(f"  MAX_INPUT_LENGTH = {int(np.percentile(all_text_lens, 85))}")
print(f"  MAX_TARGET_LENGTH = {int(np.percentile(all_sum_lens, 85))}")

print(f"\n💎 QUALITY-FIRST SETTINGS (covers 99%):")
print(f"  MAX_INPUT_LENGTH = {int(np.percentile(all_text_lens, 99))}")
print(f"  MAX_TARGET_LENGTH = {int(np.percentile(all_sum_lens, 99))}")

print(f"\n🔥 CURRENT SETTINGS:")
print(f"  MAX_INPUT_LENGTH = 384")
print(f"  MAX_TARGET_LENGTH = 128")
print(f"  Truncation rate: {np.sum(np.array(all_text_lens) > 384)/len(all_text_lens)*100:.1f}% texts, {np.sum(np.array(all_sum_lens) > 128)/len(all_sum_lens)*100:.1f}% summaries")
