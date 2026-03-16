import json

# Load first item only
with open('bansum_lte_1000_tokens.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 80)
print("BANSUM DATASET CHECK")
print("=" * 80)
print(f"\nTotal samples: {len(data)}")
print(f"\nCurrently training on: SUM1")

# Check first 3 items
for i in range(min(3, len(data))):
    item = data[i]
    print(f"\n--- Sample {i+1} ---")
    print(f"Main (input text): {len(item['main'].split())} words, ~{item.get('token_count', 'N/A')} tokens")
    print(f"Sum1: {len(item['sum1'].split())} words")
    print(f"Sum2: {len(item['sum2'].split())} words") 
    print(f"Sum3: {len(item['sum3'].split())} words")
    
print("\n" + "=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)
print("We are using: item['sum1'] as target summary")
print("MAX_INPUT_LENGTH: 850 tokens")
print("MAX_TARGET_LENGTH: 256 tokens")
