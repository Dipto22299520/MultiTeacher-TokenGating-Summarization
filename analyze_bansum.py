import json
from transformers import AutoTokenizer

print("=" * 80)
print("ANALYZING BANSUM DATASET")
print("=" * 80)

# Load data
print("\nLoading bansum_lte_1000_tokens.json...")
with open('bansum_lte_1000_tokens.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

# Load tokenizer to count tokens properly
print("\nLoading mT5 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")

# Analyze lengths
main_lengths = []
sum1_lengths = []
sum2_lengths = []
sum3_lengths = []

# Sample 5000 random items for speed
import random
sample_size = min(5000, len(data))
sampled_data = random.sample(data, sample_size)

print(f"\nAnalyzing token lengths (sampling {sample_size} items)...")
for i, item in enumerate(sampled_data):
    if i % 1000 == 0:
        print(f"  Processing {i}/{sample_size}...")
    
    main_tokens = len(tokenizer(item['main'], truncation=False)['input_ids'])
    sum1_tokens = len(tokenizer(item['sum1'], truncation=False)['input_ids'])
    sum2_tokens = len(tokenizer(item['sum2'], truncation=False)['input_ids'])
    sum3_tokens = len(tokenizer(item['sum3'], truncation=False)['input_ids'])
    
    main_lengths.append(main_tokens)
    sum1_lengths.append(sum1_tokens)
    sum2_lengths.append(sum2_tokens)
    sum3_lengths.append(sum3_tokens)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\nMAIN (Input Text):")
print(f"  Average length: {sum(main_lengths)/len(main_lengths):.1f} tokens")
print(f"  Max length: {max(main_lengths)} tokens")
print(f"  Min length: {min(main_lengths)} tokens")
print(f"  Median length: {sorted(main_lengths)[len(main_lengths)//2]} tokens")

print(f"\nSUM1 (Summary 1):")
print(f"  Average length: {sum(sum1_lengths)/len(sum1_lengths):.1f} tokens")
print(f"  Max length: {max(sum1_lengths)} tokens")
print(f"  Min length: {min(sum1_lengths)} tokens")

print(f"\nSUM2 (Summary 2):")
print(f"  Average length: {sum(sum2_lengths)/len(sum2_lengths):.1f} tokens")
print(f"  Max length: {max(sum2_lengths)} tokens")
print(f"  Min length: {min(sum2_lengths)} tokens")

print(f"\nSUM3 (Summary 3):")
print(f"  Average length: {sum(sum3_lengths)/len(sum3_lengths):.1f} tokens")
print(f"  Max length: {max(sum3_lengths)} tokens")
print(f"  Min length: {min(sum3_lengths)} tokens")

# Show coverage statistics for different max lengths
print(f"\n" + "=" * 80)
print("COVERAGE ANALYSIS FOR MAIN (Input)")
print("=" * 80)
for max_len in [512, 850, 1024]:
    coverage = sum(1 for l in main_lengths if l <= max_len) / len(main_lengths) * 100
    print(f"  Max length {max_len}: {coverage:.1f}% of samples fit")

print(f"\n" + "=" * 80)
print("CURRENT TRAINING CONFIGURATION")
print("=" * 80)
print(f"  We are training on: SUM1")
print(f"  MAX_INPUT_LENGTH: 850 tokens")
print(f"  MAX_TARGET_LENGTH: 256 tokens")
