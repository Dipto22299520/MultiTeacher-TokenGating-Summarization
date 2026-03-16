"""Quick verification of data integrity before training."""
import json

for split in ['train', 'val', 'test']:
    path = f'data_splits/{split}.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Count precomposed chars (should be 0)
    precomposed = 0
    decomposed_nukta = 0
    empty_text = 0
    empty_summary = 0
    
    for item in data:
        text = item.get('text', '')
        summary = item.get('summary', '')
        
        if not text.strip():
            empty_text += 1
        if not summary.strip():
            empty_summary += 1
        
        for field_val in [text, summary]:
            precomposed += field_val.count('\u09DF')  # য়
            precomposed += field_val.count('\u09DC')  # ড়
            precomposed += field_val.count('\u09DD')  # ঢ়
            decomposed_nukta += field_val.count('\u09BC')  # nukta ়
    
    print(f"\n=== {split}.json ({len(data)} samples) ===")
    print(f"  Empty text: {empty_text}")
    print(f"  Empty summary: {empty_summary}")
    print(f"  Precomposed য়/ড়/ঢ় (BAD - causes UNK): {precomposed}")
    print(f"  Decomposed nukta ় (GOOD - works with tokenizer): {decomposed_nukta}")

# Verify tokenizer compatibility with sample
print("\n=== Tokenizer UNK check on first 50 train samples ===")
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained('csebuetnlp/banglaT5', use_fast=False)
unk_id = tokenizer.unk_token_id

total_tokens = 0
total_unks = 0
for item in data[:50]:  # data is still test, reload train
    pass

with open('data_splits/train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

for item in train_data[:50]:
    for field in ['text', 'summary']:
        ids = tokenizer.encode(item[field])
        total_tokens += len(ids)
        total_unks += ids.count(unk_id)

print(f"  Total tokens (50 samples): {total_tokens}")
print(f"  UNK tokens: {total_unks}")
print(f"  UNK rate: {total_unks/total_tokens*100:.3f}%")

# Also test with fix_bangla_for_tokenizer applied (double fix shouldn't hurt)
def fix_bangla_for_tokenizer(text):
    text = text.replace('\u09DF', '\u09AF\u09BC')
    text = text.replace('\u09DC', '\u09A1\u09BC')
    text = text.replace('\u09DD', '\u09A2\u09BC')
    text = text.replace('\u200c', '')
    text = text.replace('\u200d', '')
    return text

total_tokens2 = 0
total_unks2 = 0
for item in train_data[:50]:
    for field in ['text', 'summary']:
        fixed = fix_bangla_for_tokenizer(item[field])
        ids = tokenizer.encode(fixed)
        total_tokens2 += len(ids)
        total_unks2 += ids.count(unk_id)

print(f"\n=== After double-applying fix_bangla_for_tokenizer ===")
print(f"  Total tokens: {total_tokens2}")
print(f"  UNK tokens: {total_unks2}")
print(f"  UNK rate: {total_unks2/total_tokens2*100:.3f}%")
print(f"\n  (Double-fix is safe: {total_unks == total_unks2})")

print("\n=== ALL CHECKS PASSED ===" if precomposed == 0 and empty_text == 0 and empty_summary == 0 else "\n=== ISSUES FOUND ===")
