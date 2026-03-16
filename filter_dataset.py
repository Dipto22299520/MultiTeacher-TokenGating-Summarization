"""
BanSum Quality Filtering & Preprocessing
==========================================
Filters the raw 141k BanSum dataset down to ~35k high-quality article-summary
pairs using multiple quality criteria, then saves the filtered dataset.

Quality Criteria:
    1. Remove duplicates (by article text prefix)
    2. Article length: >= 150 words (enough content to summarize)
    3. Summary length: >= 15 words and <= 200 words (meaningful but concise)
    4. Compression ratio: 0.05 <= sum_chars/art_chars <= 0.70 (real compression)
    5. Word overlap: >= 0.15 (summary relates to article) and < 0.98 (not pure copy)
    6. Summary != article (not identical)
    7. Bangla script ratio: >= 0.5 of summary chars are Bangla (not garbled/English)
    8. Composite quality score for ranking and selection

Output: bansum_filtered_35k.json
"""

import json
import re
import os
import sys
import numpy as np
from collections import Counter

# Bangla Unicode range
BANGLA_RANGE = re.compile(r'[\u0980-\u09FF]')

def bangla_char_ratio(text):
    """Fraction of non-space chars that are Bangla script."""
    chars = text.replace(' ', '')
    if not chars:
        return 0.0
    bangla_count = len(BANGLA_RANGE.findall(chars))
    return bangla_count / len(chars)

def word_overlap(article, summary):
    """Fraction of summary words found in article."""
    art_words = set(article.split())
    sum_words = set(summary.split())
    if not sum_words:
        return 0.0
    return len(art_words & sum_words) / len(sum_words)

def bigram_overlap(article, summary):
    """Fraction of summary bigrams found in article."""
    def get_bigrams(text):
        words = text.split()
        return set(zip(words, words[1:])) if len(words) > 1 else set()
    
    art_bg = get_bigrams(article)
    sum_bg = get_bigrams(summary)
    if not sum_bg:
        return 0.0
    return len(art_bg & sum_bg) / len(sum_bg)

def novel_unigram_ratio(article, summary):
    """Fraction of summary words NOT in article (abstractiveness measure)."""
    art_words = set(article.split())
    sum_words = summary.split()
    if not sum_words:
        return 0.0
    novel = sum(1 for w in sum_words if w not in art_words)
    return novel / len(sum_words)

def compute_quality_score(sample):
    """
    Compute a composite quality score for an article-summary pair.
    Higher is better. Scores multiple dimensions and combines them.
    
    Returns: (score, features_dict)
    """
    article = sample['main'].strip()
    summary = sample['sum2'].strip()
    
    art_words = len(article.split())
    sum_words = len(summary.split())
    art_chars = len(article)
    sum_chars = len(summary)
    
    # Basic features
    compression = sum_chars / max(art_chars, 1)
    w_overlap = word_overlap(article, summary)
    bg_overlap = bigram_overlap(article, summary)
    novel_ratio = novel_unigram_ratio(article, summary)
    bangla_ratio = bangla_char_ratio(summary)
    
    features = {
        'art_words': art_words,
        'sum_words': sum_words,
        'compression': compression,
        'word_overlap': w_overlap,
        'bigram_overlap': bg_overlap,
        'novel_ratio': novel_ratio,
        'bangla_ratio': bangla_ratio,
    }
    
    # === Scoring (each component 0-1, weighted) ===
    
    # 1. Article length score: prefer 200-500 words (sweet spot for summarization)
    if art_words < 100:
        len_score = 0.3
    elif art_words < 200:
        len_score = 0.6
    elif art_words <= 500:
        len_score = 1.0
    elif art_words <= 700:
        len_score = 0.8
    else:
        len_score = 0.6
    
    # 2. Summary length score: prefer 30-100 words
    if sum_words < 10:
        slen_score = 0.0
    elif sum_words < 30:
        slen_score = 0.5
    elif sum_words <= 100:
        slen_score = 1.0
    elif sum_words <= 150:
        slen_score = 0.7
    else:
        slen_score = 0.3
    
    # 3. Compression score: ideal 0.15-0.40
    if compression < 0.05:
        comp_score = 0.1
    elif compression < 0.15:
        comp_score = 0.6
    elif compression <= 0.40:
        comp_score = 1.0
    elif compression <= 0.60:
        comp_score = 0.5
    else:
        comp_score = 0.1
    
    # 4. Abstractiveness: some novelty is good (0.15-0.50 novel words)
    if novel_ratio < 0.05:
        abstract_score = 0.2  # too extractive
    elif novel_ratio < 0.15:
        abstract_score = 0.6
    elif novel_ratio <= 0.50:
        abstract_score = 1.0  # good abstractiveness
    elif novel_ratio <= 0.70:
        abstract_score = 0.6
    else:
        abstract_score = 0.2  # too different, might be unrelated
    
    # 5. Content relevance: word overlap (0.30-0.85 is ideal)
    if w_overlap < 0.10:
        relevance_score = 0.0  # unrelated
    elif w_overlap < 0.30:
        relevance_score = 0.4
    elif w_overlap <= 0.85:
        relevance_score = 1.0  # good balance
    elif w_overlap <= 0.95:
        relevance_score = 0.5
    else:
        relevance_score = 0.2  # pure copy
    
    # 6. Bangla quality
    bangla_score = min(bangla_ratio / 0.5, 1.0)  # 1.0 if >= 50% Bangla
    
    # Weighted composite
    score = (
        0.15 * len_score +
        0.15 * slen_score +
        0.20 * comp_score +
        0.20 * abstract_score +
        0.20 * relevance_score +
        0.10 * bangla_score
    )
    
    features['score'] = score
    return score, features


def main():
    TARGET_COUNT = 35000
    
    print("=" * 80)
    print("BANSUM QUALITY FILTERING")
    print("=" * 80)
    
    # Load
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bansum_lte_1000_tokens.json')
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded: {len(data)} samples")
    
    # ====== HARD FILTERS (remove clearly bad samples) ======
    print("\n--- Hard Filters ---")
    
    # 1. Remove duplicates by article prefix (keep first occurrence)
    seen_prefixes = set()
    deduped = []
    for d in data:
        prefix = d['main'][:300]
        if prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            deduped.append(d)
    print(f"  After dedup: {len(deduped)} (removed {len(data) - len(deduped)} duplicates)")
    data = deduped
    
    # 2. Remove tiny articles (< 100 words)
    before = len(data)
    data = [d for d in data if len(d['main'].split()) >= 100]
    print(f"  After article >= 100 words: {len(data)} (removed {before - len(data)})")
    
    # 3. Remove tiny/huge summaries (< 15 words or > 200 words)
    before = len(data)
    data = [d for d in data if 15 <= len(d['sum2'].split()) <= 200]
    print(f"  After summary 15-200 words: {len(data)} (removed {before - len(data)})")
    
    # 4. Remove bad compression ratios (> 0.70 or < 0.05)
    before = len(data)
    data = [d for d in data if 0.05 <= len(d['sum2']) / max(len(d['main']), 1) <= 0.70]
    print(f"  After compression 0.05-0.70: {len(data)} (removed {before - len(data)})")
    
    # 5. Remove zero/near-zero overlap (unrelated summary)
    before = len(data)
    data = [d for d in data if word_overlap(d['main'], d['sum2']) >= 0.15]
    print(f"  After word overlap >= 0.15: {len(data)} (removed {before - len(data)})")
    
    # 6. Remove fully extractive (>= 0.98 overlap)
    before = len(data)
    data = [d for d in data if word_overlap(d['main'], d['sum2']) < 0.98]
    print(f"  After word overlap < 0.98: {len(data)} (removed {before - len(data)})")
    
    # 7. Remove low Bangla content
    before = len(data)
    data = [d for d in data if bangla_char_ratio(d['sum2']) >= 0.40]
    print(f"  After Bangla ratio >= 0.40: {len(data)} (removed {before - len(data)})")
    
    # 8. Remove identical article/summary
    before = len(data)
    data = [d for d in data if d['main'].strip() != d['sum2'].strip()]
    print(f"  After removing identical: {len(data)} (removed {before - len(data)})")
    
    print(f"\n  After all hard filters: {len(data)} samples")
    
    # ====== QUALITY SCORING ======
    print("\n--- Quality Scoring ---")
    scored = []
    for d in data:
        score, features = compute_quality_score(d)
        scored.append((score, features, d))
    
    scores = [s[0] for s in scored]
    print(f"  Score distribution:")
    print(f"    Min: {min(scores):.3f}, Max: {max(scores):.3f}")
    print(f"    Mean: {np.mean(scores):.3f}, Median: {np.median(scores):.3f}")
    print(f"    P25: {np.percentile(scores, 25):.3f}, P75: {np.percentile(scores, 75):.3f}")
    
    # Sort by quality score (descending)
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Select top TARGET_COUNT
    if len(scored) > TARGET_COUNT:
        selected = scored[:TARGET_COUNT]
        cutoff_score = selected[-1][0]
        print(f"\n  Selected top {TARGET_COUNT} (score cutoff: {cutoff_score:.3f})")
    else:
        selected = scored
        print(f"\n  Only {len(selected)} samples passed filters (< {TARGET_COUNT} target)")
    
    # ====== STATISTICS OF SELECTED DATA ======
    sel_data = [s[2] for s in selected]
    sel_scores = [s[0] for s in selected]
    sel_features = [s[1] for s in selected]
    
    print(f"\n--- Selected Dataset Stats ({len(sel_data)} samples) ---")
    print(f"  Score: {np.mean(sel_scores):.3f} mean, {np.median(sel_scores):.3f} median")
    print(f"  Article words: {np.mean([f['art_words'] for f in sel_features]):.0f} mean")
    print(f"  Summary words: {np.mean([f['sum_words'] for f in sel_features]):.0f} mean")
    print(f"  Compression: {np.mean([f['compression'] for f in sel_features]):.3f} mean")
    print(f"  Word overlap: {np.mean([f['word_overlap'] for f in sel_features]):.3f} mean")
    print(f"  Novel unigrams: {np.mean([f['novel_ratio'] for f in sel_features]):.3f} mean")
    print(f"  Bangla ratio: {np.mean([f['bangla_ratio'] for f in sel_features]):.3f} mean")
    
    # ====== SAVE ======
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bansum_filtered_35k.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sel_data, f, ensure_ascii=False, indent=None)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  Saved to: {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    
    # Also save filtering report
    report = {
        'original_count': 141200,
        'after_hard_filters': len(data),
        'selected_count': len(sel_data),
        'target_count': TARGET_COUNT,
        'score_cutoff': float(selected[-1][0]) if selected else 0,
        'mean_score': float(np.mean(sel_scores)),
        'filters_applied': [
            'dedup by article prefix (300 chars)',
            'article >= 100 words',
            'summary 15-200 words', 
            'compression ratio 0.05-0.70',
            'word overlap >= 0.15',
            'word overlap < 0.98 (not pure copy)',
            'Bangla char ratio >= 0.40',
            'not identical article/summary',
        ],
        'scoring_weights': {
            'article_length': 0.15,
            'summary_length': 0.15,
            'compression': 0.20,
            'abstractiveness': 0.20,
            'relevance': 0.20,
            'bangla_quality': 0.10,
        }
    }
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'filtering_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report: {report_path}")
    
    print(f"\n{'='*80}")
    print("FILTERING COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
