"""
Training Data Generator for >1000 Token Articles

Processes bangla_train_gt_1000.json through the full chunking pipeline:
  1. Sentence-aligned chunking (Solution 1)
  2. Sliding overlap (Solution 2)
  3. Memory-aware headers (Solution 3)
  4. Optional [CHUNK_BOUNDARY] token insertion (Solution 4)

Each chunk becomes an independent training example paired with the FULL
reference summary. The model learns to identify which parts of the summary
relate to each chunk via attention.

Output: bangla_train_gt_1000_chunked.json
"""

import json
import os
import sys
import gc
import warnings
import logging
import argparse
import time
from typing import List, Dict, Any, Optional

from tqdm import tqdm

# Suppress the "Token indices sequence length is longer than..." warning
# We intentionally encode long texts just for counting tokens
warnings.filterwarnings("ignore", message="Token indices sequence length is longer than")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from bangla_sentence_splitter import split_sentences, count_bpe_tokens
from chunk_processor import SentenceChunker, Chunk
from memory_header import MemoryHeaderInjector, process_article_with_memory, MEMORY_PREFIX, _estimate_tokens
from attention_bias import CHUNK_BOUNDARY_TOKEN


def generate_chunked_training_data(
    input_file: str = "bangla_train_gt_1000.json",
    output_file: str = "bangla_train_gt_1000_chunked.jsonl",
    max_chunk_tokens: int = 900,
    overlap_sentences: int = 3,
    max_memory_tokens: int = 100,
    include_boundary_tokens: bool = True,
    include_single_chunk: bool = True,
    start_article: int = 0,
    limit_articles: Optional[int] = None,
    append_mode: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Process the gt_1000 dataset into chunked training examples.
    
    Args:
        input_file: Path to the raw gt_1000 JSON file
        output_file: Path for the chunked output JSON file
        max_chunk_tokens: Maximum BPE tokens per chunk
        overlap_sentences: Number of overlap sentences between chunks
        max_memory_tokens: Maximum tokens for memory headers
        include_boundary_tokens: If True, insert [CHUNK_BOUNDARY] markers in overlap zone
        include_single_chunk: If True, also include articles that fit in 1 chunk
        verbose: Print progress
    
    Returns:
        Statistics dict with counts and token distributions
    """
    # Load data
    if verbose:
        print(f"\nLoading data from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Apply start and limit
    end_article = len(data)
    if limit_articles is not None:
        end_article = min(start_article + int(limit_articles), len(data))
    
    data = data[start_article:end_article]
    
    if verbose:
        print(f"Processing articles {start_article} to {end_article-1} ({len(data)} articles)")
        print(f"\nℹ️  Data preparation uses CPU (tokenizer is CPU-bound).")
        print(f"   GPU will be used during training (train_bangla_chunked.py).\n")
        print("Pre-loading tokenizer...")
        from bangla_sentence_splitter import get_bangla_tokenizer
        get_bangla_tokenizer()  # Load once, cache for reuse
        print("✓ Tokenizer loaded\n")
    
    # Statistics
    stats = {
        'total_articles': len(data),
        'total_chunks_created': 0,
        'articles_with_1_chunk': 0,
        'articles_with_2_chunks': 0,
        'articles_with_3plus_chunks': 0,
        'skipped_empty': 0,
        'max_chunks_per_article': 0,
        'avg_chunks_per_article': 0,
        'token_distribution': {
            'original_tokens': [],
            'chunk_tokens': [],
            'memory_header_tokens': [],
        }
    }
    
    # Stream chunks to JSONL (one JSON object per line) to avoid OOM and broken JSON arrays.
    if verbose:
        print(f"\nWriting chunked output as JSONL: {output_file}")

    chunks_written = 0

    try:
        mode = 'a' if append_mode else 'w'
        with open(output_file, mode, encoding='utf-8') as output_handle:
            article_iter = tqdm(
                data,
                desc="Chunking articles",
                unit="article",
                initial=0,
                total=len(data),
                mininterval=0.5,
                dynamic_ncols=True,
            ) if verbose else data

            for local_idx, article in enumerate(article_iter):
                article_idx = start_article + local_idx
                
                # Progress indicator every 50 articles
                if verbose and article_idx % 50 == 0 and article_idx > 0:
                    print(f"\n📍 Article {article_idx}...", flush=True)
                
                # Print current article being processed (helps identify hangs)
                if verbose and article_idx % 10 == 0:
                    sys.stdout.flush()
                
                article_start = time.time()
                text = article.get('text', '').strip()
                summary = article.get('summary', '').strip()

                if not text or not summary:
                    stats['skipped_empty'] += 1
                    continue

                # Use fast estimate for stats (saves tokenizer overhead)
                original_tokens = _estimate_tokens(text)
                stats['token_distribution']['original_tokens'].append(original_tokens)

                # SKIP known problematic articles that cause infinite loops
                # These have pathological overlap patterns or single-giant-sentence issues
                SKIP_ARTICLES = [400]  # Add more as discovered
                if article_idx in SKIP_ARTICLES:
                    print(f"\n⚠️  Skipping article {article_idx} (known infinite loop)", flush=True)
                    stats['skipped_empty'] += 1
                    continue

                # Show article size
                if len(text) > 15000 and verbose:
                    print(f"\n⚠️  Article {article_idx}: {len(text)} chars (large, may be slow)", flush=True)

                # Run pipeline with timeout check
                pipeline_start = time.time()
                memory_chunks = process_article_with_memory(
                    text,
                    max_tokens=max_chunk_tokens,
                    overlap_sentences=overlap_sentences,
                    max_memory_tokens=max_memory_tokens,
                    verbose=False  # Disable detailed logging for normal runs
                )
                pipeline_elapsed = time.time() - pipeline_start
                
                # If chunking took >15s, it's likely stuck in infinite loop - skip it
                if pipeline_elapsed > 15.0:
                    print(f"\n🚨 Article {article_idx} TIMEOUT ({pipeline_elapsed:.1f}s) - will skip on next run!", flush=True)
                    print(f"   Add {article_idx} to SKIP_ARTICLES list", flush=True)
                    # Continue this time but warn user

                if not memory_chunks:
                    stats['skipped_empty'] += 1
                    if verbose:
                        print(f"    ⚠️ Skipped (empty chunks)", flush=True)
                    continue

                num_chunks = len(memory_chunks)
                article_elapsed = time.time() - article_start
                
                # Warn if article took abnormally long
                if article_elapsed > 10.0 and verbose:
                    print(f"\n⏱️  Article {article_idx} took {article_elapsed:.1f}s", flush=True)

                # Track article chunk stats
                if num_chunks == 1:
                    stats['articles_with_1_chunk'] += 1
                    if not include_single_chunk:
                        continue
                elif num_chunks == 2:
                    stats['articles_with_2_chunks'] += 1
                else:
                    stats['articles_with_3plus_chunks'] += 1

                stats['max_chunks_per_article'] = max(stats['max_chunks_per_article'], num_chunks)
                stats['total_chunks_created'] += num_chunks

                # Emit one training example per chunk
                for mc in memory_chunks:
                    chunk_text = mc.full_text
                    if include_boundary_tokens and mc.has_memory and mc.memory_header:
                        chunk_text = (
                            f"[{MEMORY_PREFIX} {mc.memory_header}] "
                            f"{CHUNK_BOUNDARY_TOKEN} "
                            f"{mc.original_chunk.text}"
                        )

                    chunk_tokens = _estimate_tokens(chunk_text)
                    memory_tokens = _estimate_tokens(mc.memory_header) if mc.memory_header else 0

                    stats['token_distribution']['chunk_tokens'].append(chunk_tokens)
                    stats['token_distribution']['memory_header_tokens'].append(memory_tokens)

                    training_example = {
                        'text': chunk_text,
                        'summary': summary,
                        'chunk_index': mc.chunk_index,
                        'total_chunks': mc.total_chunks,
                        'article_id': article_idx,
                        'original_tokens': original_tokens,
                        'chunk_tokens': chunk_tokens,
                        'has_memory': mc.has_memory,
                        'memory_tokens': memory_tokens,
                    }

                    output_handle.write(json.dumps(training_example, ensure_ascii=False) + "\n")
                    chunks_written += 1

                # Force flush
                output_handle.flush()
                sys.stdout.flush()

                if verbose and hasattr(article_iter, 'set_postfix'):
                    article_iter.set_postfix(
                        chunks_written=chunks_written,
                        this_article=num_chunks,
                    )
                
                # Progress checkpoint every 100 articles
                if verbose and (article_idx + 1) % 100 == 0:
                    print(f"\n✓ Checkpoint: {article_idx + 1} articles processed, {chunks_written} chunks written", flush=True)

                if (article_idx + 1) % 50 == 0:
                    gc.collect()

    except Exception as e:
        print(f"\n❌ Error during chunking: {e}")
        raise
    
    if verbose:
        print(f"\n✓ Saved {chunks_written} chunk lines to: {output_file}")
    
    # Compute averages
    if stats['total_articles'] - stats['skipped_empty'] > 0:
        valid_articles = stats['total_articles'] - stats['skipped_empty']
        stats['avg_chunks_per_article'] = stats['total_chunks_created'] / valid_articles
    
    # Remove raw lists from stats (for JSON serialization), compute summaries
    token_stats = stats.pop('token_distribution')
    if token_stats['original_tokens']:
        import numpy as np
        stats['original_token_stats'] = {
            'min': int(min(token_stats['original_tokens'])),
            'max': int(max(token_stats['original_tokens'])),
            'mean': float(np.mean(token_stats['original_tokens'])),
            'median': float(np.median(token_stats['original_tokens'])),
        }
    if token_stats['chunk_tokens']:
        import numpy as np
        stats['chunk_token_stats'] = {
            'min': int(min(token_stats['chunk_tokens'])),
            'max': int(max(token_stats['chunk_tokens'])),
            'mean': float(np.mean(token_stats['chunk_tokens'])),
            'median': float(np.median(token_stats['chunk_tokens'])),
        }
    
    # Save stats
    stats_file = output_file.replace('.json', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"\n{'='*60}")
        print("CHUNKING STATISTICS")
        print(f"{'='*60}")
        print(f"Total articles processed: {stats['total_articles']}")
        print(f"Skipped (empty): {stats['skipped_empty']}")
        print(f"Total training chunks created: {stats['total_chunks_created']}")
        print(f"  1-chunk articles: {stats['articles_with_1_chunk']}")
        print(f"  2-chunk articles: {stats['articles_with_2_chunks']}")
        print(f"  3+ chunk articles: {stats['articles_with_3plus_chunks']}")
        print(f"Max chunks per article: {stats['max_chunks_per_article']}")
        print(f"Avg chunks per article: {stats['avg_chunks_per_article']:.1f}")
        if 'original_token_stats' in stats:
            ots = stats['original_token_stats']
            print(f"\nOriginal article tokens: "
                  f"min={ots['min']}, max={ots['max']}, "
                  f"mean={ots['mean']:.0f}, median={ots['median']:.0f}")
        if 'chunk_token_stats' in stats:
            cts = stats['chunk_token_stats']
            print(f"Chunk tokens: "
                  f"min={cts['min']}, max={cts['max']}, "
                  f"mean={cts['mean']:.0f}, median={cts['median']:.0f}")
        print(f"\nFiles created:")
        print(f"  {output_file}")
        print(f"  {stats_file}")
    
    return stats


def generate_combined_training_data(
    lte_file: str = "bangla_train_lte_1000.json",
    chunked_file: str = "bangla_train_gt_1000_chunked.jsonl",
    output_file: str = "bangla_train_combined.jsonl",
    verbose: bool = True
) -> int:
    """
    Combine the ≤1000 token data with the chunked >1000 token data
    into a single training file.
    
    This prevents catastrophic forgetting: the model trains on both
    short (original) and long (chunked) articles simultaneously.
    
    Args:
        lte_file: ≤1000 token data file
        chunked_file: Chunked >1000 token data file
        output_file: Combined output file
        verbose: Print progress
    
    Returns:
        Total number of training examples in combined file
    """
    if verbose:
        print(f"\nCombining training data:")
        print(f"  ≤1000 tokens: {lte_file}")
        print(f"  >1000 chunked: {chunked_file}")
    
    # Stream-combine to JSONL to avoid loading both datasets into RAM.
    total_written = 0

    with open(output_file, 'w', encoding='utf-8') as out:
        with open(lte_file, 'r', encoding='utf-8') as f:
            lte_data = json.load(f)
        if verbose:
            print(f"\n  ≤1000 examples: {len(lte_data)}")

        for item in lte_data:
            out.write(json.dumps({
                'text': item['text'],
                'summary': item['summary'],
                'chunk_index': 0,
                'total_chunks': 1,
                'article_id': -1,
                'original_tokens': -1,
                'chunk_tokens': -1,
                'has_memory': False,
                'memory_tokens': 0,
                'source': 'lte_1000',
            }, ensure_ascii=False) + "\n")
            total_written += 1

        # Append chunked lines
        chunked_count = 0
        with open(chunked_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                obj['source'] = 'gt_1000_chunked'
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                total_written += 1
                chunked_count += 1

        if verbose:
            print(f"  >1000 chunked examples: {chunked_count}")
            print(f"  Combined total: {total_written}")
            print(f"\n  Saved to: {output_file}")

    return total_written


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare chunked training data for gt_1000")
    parser.add_argument("--start", type=int, default=0, help="Start from article N (for incremental processing)")
    parser.add_argument("--limit", type=int, default=None, help="Only process N articles from start (debug)")
    parser.add_argument("--append", action="store_true", help="Append to existing output file instead of overwriting")
    parser.add_argument("--no-boundary", action="store_true", help="Disable [CHUNK_BOUNDARY] token insertion")
    parser.add_argument("--out-chunked", type=str, default="bangla_train_gt_1000_chunked.jsonl")
    parser.add_argument("--out-combined", type=str, default="bangla_train_combined.jsonl")
    parser.add_argument("--skip-combine", action="store_true", help="Skip combining with lte_1000 data")
    args = parser.parse_args()

    print("=" * 60)
    print("GENERATING CHUNKED TRAINING DATA (JSONL STREAMING)")
    print("=" * 60)

    generate_chunked_training_data(
        input_file="bangla_train_gt_1000.json",
        output_file=args.out_chunked,
        max_chunk_tokens=900,
        overlap_sentences=3,
        max_memory_tokens=100,
        include_boundary_tokens=not args.no_boundary,
        include_single_chunk=True,
        start_article=args.start,
        limit_articles=args.limit,
        append_mode=args.append,
        verbose=True,
    )

    if not args.skip_combine:
        print("\n" + "=" * 60)
        print("COMBINING WITH LTE_1000 DATA")
        print("=" * 60)

        total = generate_combined_training_data(
            lte_file="bangla_train_lte_1000.json",
            chunked_file=args.out_chunked,
            output_file=args.out_combined,
            verbose=True,
        )

        print(f"\n✅ Done! Total combined training examples: {total}")
    else:
        print(f"\n✅ Done! Chunked data saved to {args.out_chunked}")
