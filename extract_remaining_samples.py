import json

# Read the full test set
with open('data_splits/test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# Extract the last 388 samples (indices 1600-1987)
remaining_samples = test_data[1600:]

print(f"Total test samples: {len(test_data)}")
print(f"Extracted samples: {len(remaining_samples)}")
print(f"First sample index would be: 1600")
print(f"Last sample index would be: {1600 + len(remaining_samples) - 1}")

# Save to new file
with open('test_remaining_388.json', 'w', encoding='utf-8') as f:
    json.dump(remaining_samples, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(remaining_samples)} samples to test_remaining_388.json")
