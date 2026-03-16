"""Quick dataset analysis for quality filtering."""
import json
import numpy as np

with open('../bansum_lte_1000_tokens.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'Total samples: {len(data)}')
print(f'Keys: {list(data[0].keys())}')
print()

art_lens = [len(d['main']) for d in data]
sum_lens = [len(d['sum2']) for d in data]
art_words = [len(d['main'].split()) for d in data]
sum_words = [len(d['sum2'].split()) for d in data]

print('=== Article (chars) ===')
print(f'  Min: {min(art_lens)}, Max: {max(art_lens)}, Mean: {np.mean(art_lens):.0f}, Median: {np.median(art_lens):.0f}')
print(f'  P5: {np.percentile(art_lens, 5):.0f}, P25: {np.percentile(art_lens, 25):.0f}, P75: {np.percentile(art_lens, 75):.0f}, P95: {np.percentile(art_lens, 95):.0f}')

print('=== Article (words) ===')
print(f'  Min: {min(art_words)}, Max: {max(art_words)}, Mean: {np.mean(art_words):.0f}, Median: {np.median(art_words):.0f}')

print('=== Summary (chars) ===')
print(f'  Min: {min(sum_lens)}, Max: {max(sum_lens)}, Mean: {np.mean(sum_lens):.0f}, Median: {np.median(sum_lens):.0f}')
print(f'  P5: {np.percentile(sum_lens, 5):.0f}, P25: {np.percentile(sum_lens, 25):.0f}, P75: {np.percentile(sum_lens, 75):.0f}, P95: {np.percentile(sum_lens, 95):.0f}')

print('=== Summary (words) ===')
print(f'  Min: {min(sum_words)}, Max: {max(sum_words)}, Mean: {np.mean(sum_words):.0f}, Median: {np.median(sum_words):.0f}')

ratios = [len(d['sum2']) / max(len(d['main']), 1) for d in data]
print('=== Compression Ratio (sum_chars/art_chars) ===')
print(f'  Min: {min(ratios):.3f}, Max: {max(ratios):.3f}, Mean: {np.mean(ratios):.3f}, Median: {np.median(ratios):.3f}')

# Check for empty/tiny
empty_art = sum(1 for d in data if len(d['main'].strip()) < 50)
empty_sum = sum(1 for d in data if len(d['sum2'].strip()) < 20)
print(f'\nTiny articles (<50 chars): {empty_art}')
print(f'Tiny summaries (<20 chars): {empty_sum}')

# Check for duplicates
unique_arts = len(set(d['main'][:200] for d in data))
unique_sums = len(set(d['sum2'][:100] for d in data))
print(f'Unique article prefixes: {unique_arts} / {len(data)}')
print(f'Unique summary prefixes: {unique_sums} / {len(data)}')

# Word overlap (crude extractiveness measure)
def word_overlap(article, summary):
    art_words = set(article.split())
    sum_words = set(summary.split())
    if not sum_words:
        return 0.0
    return len(art_words & sum_words) / len(sum_words)

overlaps = [word_overlap(d['main'], d['sum2']) for d in data]
print(f'\n=== Word Overlap (summary words found in article) ===')
print(f'  Min: {min(overlaps):.3f}, Max: {max(overlaps):.3f}, Mean: {np.mean(overlaps):.3f}, Median: {np.median(overlaps):.3f}')
print(f'  Zero overlap (summary has NO words from article): {sum(1 for o in overlaps if o == 0)}')
print(f'  100% overlap (fully extractive): {sum(1 for o in overlaps if o >= 1.0)}')

# Summary == article check
identical = sum(1 for d in data if d['main'].strip() == d['sum2'].strip())
print(f'  Summary identical to article: {identical}')

# Sample
print('\n=== Sample ===')
idx = 100
print(f'Article ({len(data[idx]["main"])} chars, {len(data[idx]["main"].split())} words):')
print(f'  {data[idx]["main"][:300]}...')
print(f'Summary ({len(data[idx]["sum2"])} chars, {len(data[idx]["sum2"].split())} words):')
print(f'  {data[idx]["sum2"][:300]}...')
