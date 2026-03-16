import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import re

print("="*80)
print("DATA QUALITY ANALYSIS")
print("="*80)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")

# Load datasets
print("\nLoading datasets...")
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
test_df = pd.read_csv("data/test.csv")

all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
print(f"Total samples: {len(all_df)} (train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)})")

# ============================================================================
# 1. CHECK FOR DUPLICATES
# ============================================================================
print("\n" + "="*80)
print("1. DUPLICATE CHECK")
print("="*80)

duplicate_texts = all_df['text'].duplicated().sum()
duplicate_summaries = all_df['summary'].duplicated().sum()
duplicate_pairs = all_df[['text', 'summary']].duplicated().sum()

print(f"Duplicate texts: {duplicate_texts} ({duplicate_texts/len(all_df)*100:.2f}%)")
print(f"Duplicate summaries: {duplicate_summaries} ({duplicate_summaries/len(all_df)*100:.2f}%)")
print(f"Duplicate pairs: {duplicate_pairs} ({duplicate_pairs/len(all_df)*100:.2f}%)")

# ============================================================================
# 2. CHECK FOR MISSING/MALFORMED DATA
# ============================================================================
print("\n" + "="*80)
print("2. MISSING/MALFORMED DATA CHECK")
print("="*80)

missing_text = all_df['text'].isna().sum()
missing_summary = all_df['summary'].isna().sum()
empty_text = (all_df['text'].str.strip() == '').sum()
empty_summary = (all_df['summary'].str.strip() == '').sum()

print(f"Missing text: {missing_text}")
print(f"Missing summary: {missing_summary}")
print(f"Empty text: {empty_text}")
print(f"Empty summary: {empty_summary}")

# ============================================================================
# 3. LENGTH RATIO ANALYSIS (detect problematic samples)
# ============================================================================
print("\n" + "="*80)
print("3. LENGTH RATIO ANALYSIS")
print("="*80)

# Calculate length ratios
text_lens = []
summary_lens = []
compression_ratios = []

for idx, row in all_df.iterrows():
    text = str(row['text'])
    summary = str(row['summary'])
    
    text_tokens = tokenizer(text, truncation=False)['input_ids']
    summary_tokens = tokenizer(summary, truncation=False)['input_ids']
    
    text_len = len(text_tokens)
    summary_len = len(summary_tokens)
    
    text_lens.append(text_len)
    summary_lens.append(summary_len)
    
    if text_len > 0:
        compression_ratios.append(text_len / summary_len)
    
    if idx >= 5000:  # Sample 5000 for speed
        break

print(f"\nCompression ratio (text_len / summary_len):")
print(f"  Mean: {np.mean(compression_ratios):.1f}x")
print(f"  Median: {np.median(compression_ratios):.1f}x")
print(f"  10th percentile: {np.percentile(compression_ratios, 10):.1f}x")
print(f"  90th percentile: {np.percentile(compression_ratios, 90):.1f}x")

# Flag suspicious samples
suspicious_low = np.sum(np.array(compression_ratios) < 5)  # Summary almost as long as text
suspicious_high = np.sum(np.array(compression_ratios) > 100)  # Summary way too short

print(f"\n⚠️  Suspicious samples:")
print(f"  Too little compression (<5x): {suspicious_low} ({suspicious_low/len(compression_ratios)*100:.2f}%)")
print(f"  Too much compression (>100x): {suspicious_high} ({suspicious_high/len(compression_ratios)*100:.2f}%)")

# ============================================================================
# 4. SUMMARY QUALITY CHECKS
# ============================================================================
print("\n" + "="*80)
print("4. SUMMARY QUALITY CHECKS")
print("="*80)

# Check for extractive summaries (exact copy of text)
extractive_count = 0
for idx, row in all_df.iterrows():
    text = str(row['text']).lower()
    summary = str(row['summary']).lower()
    
    # If summary is exactly in text (might be extractive)
    if summary in text and len(summary) > 10:
        extractive_count += 1
    
    if idx >= 5000:
        break

print(f"Potentially extractive summaries: {extractive_count} ({extractive_count/5000*100:.2f}%)")
print("  Note: Extractive is OK for Bangla summarization")

# Check for very short summaries (might be low quality)
very_short = sum(1 for s in summary_lens if s < 10)
print(f"\nVery short summaries (<10 tokens): {very_short} ({very_short/len(summary_lens)*100:.2f}%)")

# Check for very long summaries (might not be summaries)
very_long = sum(1 for s in summary_lens if s > 50)
print(f"Very long summaries (>50 tokens): {very_long} ({very_long/len(summary_lens)*100:.2f}%)")

# ============================================================================
# 5. SHOW SAMPLE OUTLIERS
# ============================================================================
print("\n" + "="*80)
print("5. SAMPLE OUTLIERS")
print("="*80)

# Get indices of extreme compression ratios
ratio_arr = np.array(compression_ratios)
low_idx = np.where(ratio_arr < 5)[0]
high_idx = np.where(ratio_arr > 100)[0]

if len(low_idx) > 0:
    print("\n❌ EXAMPLE: Low compression (summary too long):")
    idx = low_idx[0]
    print(f"   Text (first 200 chars): {all_df.iloc[idx]['text'][:200]}...")
    print(f"   Summary: {all_df.iloc[idx]['summary']}")
    print(f"   Ratio: {ratio_arr[idx]:.1f}x")

if len(high_idx) > 0:
    print("\n❌ EXAMPLE: High compression (summary too short):")
    idx = high_idx[0]
    print(f"   Text (first 200 chars): {all_df.iloc[idx]['text'][:200]}...")
    print(f"   Summary: {all_df.iloc[idx]['summary']}")
    print(f"   Ratio: {ratio_arr[idx]:.1f}x")

# ============================================================================
# 6. RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("6. RECOMMENDATIONS")
print("="*80)

issues_found = []
total_bad_samples = 0

if duplicate_pairs > 0:
    issues_found.append(f"Remove {duplicate_pairs} duplicate pairs")
    total_bad_samples += duplicate_pairs

if missing_text + missing_summary + empty_text + empty_summary > 0:
    bad = missing_text + missing_summary + empty_text + empty_summary
    issues_found.append(f"Remove {bad} missing/empty samples")
    total_bad_samples += bad

# Estimate bad samples from compression ratio
estimated_bad_compression = int((suspicious_low + suspicious_high) / 5000 * len(all_df))
if estimated_bad_compression > 0:
    issues_found.append(f"~{estimated_bad_compression} samples with suspicious compression ratios")
    total_bad_samples += estimated_bad_compression

if len(issues_found) > 0:
    print("\n⚠️  ISSUES FOUND:")
    for issue in issues_found:
        print(f"   - {issue}")
    
    print(f"\nEstimated bad samples: ~{total_bad_samples} ({total_bad_samples/len(all_df)*100:.1f}% of dataset)")
    print(f"After cleaning: ~{len(all_df) - total_bad_samples} good samples")
    
    print("\n📋 NEXT STEPS:")
    print("   1. Run data_cleaning.py to remove bad samples")
    print("   2. Retrain with cleaned dataset")
    print(f"   3. Training will be faster with {total_bad_samples} fewer samples")
else:
    print("\n✅ DATA LOOKS GOOD!")
    print("   No major quality issues detected.")
    print("   Safe to proceed with training on all 80k samples.")

# ============================================================================
# 7. QUALITY SCORE SUMMARY
# ============================================================================
print("\n" + "="*80)
print("7. OVERALL DATA QUALITY SCORE")
print("="*80)

quality_score = 100
deductions = []

if duplicate_pairs > 0:
    dup_penalty = min(10, duplicate_pairs / len(all_df) * 100)
    quality_score -= dup_penalty
    deductions.append(f"-{dup_penalty:.1f} for duplicates")

if (missing_text + missing_summary + empty_text + empty_summary) > 0:
    missing_penalty = min(10, (missing_text + missing_summary) / len(all_df) * 100)
    quality_score -= missing_penalty
    deductions.append(f"-{missing_penalty:.1f} for missing data")

if suspicious_low / len(compression_ratios) > 0.05:
    quality_score -= 15
    deductions.append("-15 for low compression samples")

if suspicious_high / len(compression_ratios) > 0.05:
    quality_score -= 15
    deductions.append("-15 for high compression samples")

print(f"\nQuality Score: {quality_score:.1f}/100")
if deductions:
    print("Deductions:")
    for d in deductions:
        print(f"  {d}")

if quality_score >= 90:
    print("\n✅ EXCELLENT - Proceed with training")
elif quality_score >= 75:
    print("\n⚠️  GOOD - Training should work but consider cleaning")
elif quality_score >= 60:
    print("\n⚠️  FAIR - Recommend cleaning before training")
else:
    print("\n❌ POOR - Must clean data before training")
