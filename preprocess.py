"""
Data Preprocessing for Bangla Summarization
Cleans data, removes nulls, and splits into train/val/test (70/20/10)
"""

import pandas as pd
import os

print("=" * 60)
print("DATA PREPROCESSING - BANGLA SUMMARIZATION")
print("=" * 60)

# Load dataset
print("\n[1/4] Loading dataset...")
df = pd.read_csv('text_summarization.csv')
print(f"Total samples: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Check for nulls
print("\n[2/4] Checking for null values...")
null_counts = df.isnull().sum()
print(null_counts)

# Clean data
print("\n[3/4] Cleaning data...")
initial_count = len(df)
df = df.dropna()  # Remove null values
df = df[df['text'].str.strip() != '']  # Remove empty text
df = df[df['summary'].str.strip() != '']  # Remove empty summaries
df = df.reset_index(drop=True)
cleaned_count = len(df)
removed_count = initial_count - cleaned_count

print(f"Removed {removed_count} invalid samples")
print(f"Clean samples: {cleaned_count}")

# Split data: 70% train, 20% val, 10% test
print("\n[4/4] Splitting data (70/20/10)...")
train_size = int(0.7 * len(df))
val_size = int(0.2 * len(df))
test_size = len(df) - train_size - val_size

train_df = df[:train_size]
val_df = df[train_size:train_size + val_size]
test_df = df[train_size + val_size:]

print(f"Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
print(f"Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

# Save splits
print("\n[5/5] Saving splits...")
os.makedirs('data', exist_ok=True)

train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print(f"\n✓ Saved to data/train.csv")
print(f"✓ Saved to data/val.csv")
print(f"✓ Saved to data/test.csv")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE!")
print("=" * 60)
print(f"\nDataset Statistics:")
print(f"  Total clean samples: {cleaned_count:,}")
print(f"  Train samples: {len(train_df):,}")
print(f"  Validation samples: {len(val_df):,}")
print(f"  Test samples: {len(test_df):,}")
print(f"\nReady for model training!")