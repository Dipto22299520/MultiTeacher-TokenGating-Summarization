import json

t = json.load(open('reduce_data/reduce_train.json', 'r', encoding='utf-8'))
v = json.load(open('reduce_data/reduce_val.json', 'r', encoding='utf-8'))
te = json.load(open('reduce_data/reduce_test.json', 'r', encoding='utf-8'))

print(f"Train: {len(t):,}")
print(f"Val: {len(v):,}")
print(f"Test: {len(te):,}")
print(f"Total: {len(t)+len(v)+len(te):,}")

augs = {}
for s in t:
    a = s.get("augmentation", "clean")
    augs[a] = augs.get(a, 0) + 1
print(f"Train augmentation: {augs}")

print(f"Sample keys: {list(t[0].keys())}")
print(f"Sample text[:200]: {t[0]['text'][:200]}")
print(f"Sample summary[:200]: {t[0]['summary'][:200]}")
