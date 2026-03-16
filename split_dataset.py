"""
Split bangla_train_combined.json into train/val/test sets
"""
import json
import os
import random
from pathlib import Path

# Configuration
INPUT_FILE = "bangla_train_combined.json"
OUTPUT_DIR = "data_splits"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42


def fix_bangla_for_tokenizer(text):
    """
    Decompose precomposed Bangla chars that BanglaT5 tokenizer maps to <unk>.
    
    য় (U+09DF) -> য + ় (U+09AF + U+09BC)
    ড় (U+09DC) -> ড + ় (U+09A1 + U+09BC)
    ঢ় (U+09DD) -> ঢ + ় (U+09A2 + U+09BC)
    """
    text = text.replace('\u09DF', '\u09AF\u09BC')
    text = text.replace('\u09DC', '\u09A1\u09BC')
    text = text.replace('\u09DD', '\u09A2\u09BC')
    text = text.replace('\u2018', "'")
    text = text.replace('\u2019', "'")
    text = text.replace('\u201C', '"')
    text = text.replace('\u201D', '"')
    text = text.replace('\u2014', '-')
    text = text.replace('\u2013', '-')
    text = text.replace('\u2026', '...')
    text = text.replace('\u200c', '')
    text = text.replace('\u200d', '')
    return text

def split_dataset():
    """Split dataset into train/val/test and save to separate folder"""
    
    print("="*80)
    print("DATASET SPLITTING")
    print("="*80)
    
    # Load the combined dataset
    print(f"\n📂 Loading dataset: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_samples = len(data)
    print(f"   ✅ Total samples loaded: {total_samples:,}")
    
    # Shuffle data with fixed seed for reproducibility
    random.seed(RANDOM_SEED)
    random.shuffle(data)
    print(f"   ✅ Data shuffled (seed={RANDOM_SEED})")
    
    # Fix Bangla text for BanglaT5 tokenizer compatibility
    print(f"   🔧 Fixing Bangla text (decomposing য়/ড়/ঢ় for tokenizer)...")
    for item in data:
        item['text'] = fix_bangla_for_tokenizer(item['text'])
        item['summary'] = fix_bangla_for_tokenizer(item['summary'])
    print(f"   ✅ Text normalization complete")
    
    # Calculate split sizes
    train_size = int(total_samples * TRAIN_RATIO)
    val_size = int(total_samples * VAL_RATIO)
    test_size = total_samples - train_size - val_size
    
    print(f"\n📊 Split sizes:")
    print(f"   Train: {train_size:,} ({TRAIN_RATIO*100:.0f}%)")
    print(f"   Val:   {val_size:,} ({VAL_RATIO*100:.0f}%)")
    print(f"   Test:  {test_size:,} ({TEST_RATIO*100:.0f}%)")
    
    # Split the data
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    print(f"\n📁 Created output directory: {OUTPUT_DIR}/")
    
    # Save splits
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    print(f"\n💾 Saving split files...")
    for split_name, split_data in splits.items():
        output_file = output_path / f"{split_name}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"   ✅ {split_name}.json - {len(split_data):,} samples ({file_size:.1f} MB)")
    
    # Validate splits
    print(f"\n✅ Validation:")
    print(f"   Total original: {total_samples:,}")
    print(f"   Total split:    {len(train_data) + len(val_data) + len(test_data):,}")
    print(f"   Match: {'✅' if total_samples == len(train_data) + len(val_data) + len(test_data) else '❌'}")
    
    # Show sample statistics
    print(f"\n📈 Sample text/summary length statistics:")
    for split_name, split_data in splits.items():
        text_lengths = [len(str(s['text'])) for s in split_data[:1000]]
        summary_lengths = [len(str(s['summary'])) for s in split_data[:1000]]
        
        avg_text = sum(text_lengths) / len(text_lengths)
        avg_summary = sum(summary_lengths) / len(summary_lengths)
        
        print(f"   {split_name.capitalize():5s} - Avg text: {avg_text:,.0f} chars, Avg summary: {avg_summary:,.0f} chars")
    
    print(f"\n{'='*80}")
    print(f"✅ Dataset split complete!")
    print(f"{'='*80}")
    print(f"\nℹ️  Files saved in: {OUTPUT_DIR}/")
    print(f"   - train.json")
    print(f"   - val.json")
    print(f"   - test.json")
    print(f"\n💡 You can now use these files for training:")
    print(f"   python train_full_document.py --train_file data_splits/train.json --val_file data_splits/val.json")

if __name__ == "__main__":
    try:
        split_dataset()
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_FILE} not found!")
        print(f"   Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
