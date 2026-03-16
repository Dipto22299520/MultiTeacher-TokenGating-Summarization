"""
Chunked Summarization Pipeline — MapReduce with Salience Model

Uses the trained BanglaT5 (checkpoint-7000) as the salience model:
  1. CHUNK: Split long article into sentence-aligned ~900-token chunks
  2. MAP:   Summarize each chunk independently (model learned what's important)
  3. REDUCE: Concatenate chunk summaries → summarize again for coherence

The model was trained on full documents (first 1024 tokens → gold summary),
so it learned SALIENCE RANKING: what facts matter, what to keep, what to drop.
When applied to chunks, each chunk summary captures the locally important info.
The reduce pass then merges and deduplicates.

Usage:
    python run_pipeline.py                           # Test on 10 articles
    python run_pipeline.py --text "long article..."  # Single article
    python run_pipeline.py --file article.txt        # From file
"""

import json
import sys
import argparse
import torch
from typing import List, Tuple

from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
from bangla_sentence_splitter import split_sentences, count_bpe_tokens
from chunk_processor import SentenceChunker


def fix_bangla_for_tokenizer(text):
    """Decompose precomposed Bangla chars that cause <unk> in BanglaT5."""
    text = text.replace('\u09DF', '\u09AF\u09BC')  # য় -> য + ়
    text = text.replace('\u09DC', '\u09A1\u09BC')  # ড় -> ড + ়
    text = text.replace('\u09DD', '\u09A2\u09BC')  # ঢ় -> ঢ + ়
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201C', '"').replace('\u201D', '"')
    text = text.replace('\u2014', '-').replace('\u2013', '-')
    text = text.replace('\u2026', '...')
    text = text.replace('\u200c', '').replace('\u200d', '')
    return text


class ChunkedSummarizer:
    """
    MapReduce summarization pipeline with separate MAP and REDUCE models.
    
    MAP model  (Step 1): Summarizes individual chunks (trained on full articles)
    REDUCE model (Step 2): Merges chunk summaries into final summary
    
    Args:
        map_model_path: Path to MAP checkpoint (Step 1 — salience)
        reduce_model_path: Path to REDUCE checkpoint (Step 2 — merging)
        max_chunk_tokens: Max tokens per chunk (default 900)
        overlap_sentences: Sentence overlap between chunks (default 3)
        device: 'cuda' or 'cpu'
    """
    
    def __init__(
        self,
        map_model_path: str = "./banglaT5_full_doc_20260215_123349/checkpoint-7000",
        reduce_model_path: str = "./banglaT5_reduce_task_20260217_111025/checkpoint-6000",
        max_chunk_tokens: int = 900,
        overlap_sentences: int = 3,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_chunk_tokens = max_chunk_tokens
        self.map_prefix = "summarize bangla news: "
        self.reduce_prefix = "summarize multiple summaries: "
        
        # Load MAP model (Step 1 — chunk summarization) in BF16
        print(f"Loading MAP model from {map_model_path}...")
        self.map_tokenizer = T5Tokenizer.from_pretrained(map_model_path, use_fast=False)
        self.map_model = AutoModelForSeq2SeqLM.from_pretrained(
            map_model_path, torch_dtype=torch.bfloat16
        )
        self.map_model.to(self.device)
        self.map_model.eval()
        print(f"  MAP model loaded on {self.device} (bf16)")
        
        # Load REDUCE model (Step 2 — merge chunk summaries) in BF16
        print(f"Loading REDUCE model from {reduce_model_path}...")
        self.reduce_tokenizer = T5Tokenizer.from_pretrained(reduce_model_path, use_fast=False)
        self.reduce_model = AutoModelForSeq2SeqLM.from_pretrained(
            reduce_model_path, torch_dtype=torch.bfloat16
        )
        self.reduce_model.to(self.device)
        self.reduce_model.eval()
        print(f"  REDUCE model loaded on {self.device} (bf16)")
        
        # Chunker
        self.chunker = SentenceChunker(
            max_tokens=max_chunk_tokens,
            overlap_sentences=overlap_sentences
        )
        
        # Generation config for MAP (chunk summaries — can be shorter)
        self.map_gen_kwargs = {
            "max_length": 256,
            "min_length": 40,
            "num_beams": 5,
            "length_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.0,
            "early_stopping": True,
        }
        
        # Generation config for REDUCE (final summary — must be longer to match references)
        self.reduce_gen_kwargs = {
            "max_length": 256,
            "min_length": 80,
            "num_beams": 5,
            "length_penalty": 2.0,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.0,
            "early_stopping": True,
        }
        
        # Single-pass config (direct summarization — full article)
        self.single_gen_kwargs = {
            "max_length": 256,
            "min_length": 60,
            "num_beams": 5,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.0,
            "early_stopping": True,
        }
    
    @torch.no_grad()
    def summarize(self, text: str, verbose: bool = False) -> dict:
        """
        Summarize any-length Bangla article.
        
        Returns dict with:
            summary: Final summary text
            chunk_summaries: List of per-chunk summaries
            num_chunks: Number of chunks
            method: 'single_pass' or 'map_reduce'
        """
        # Fix Bangla encoding
        text = fix_bangla_for_tokenizer(text)
        
        total_tokens = count_bpe_tokens(text)
        
        if verbose:
            print(f"  Input: {total_tokens} tokens")
        
        # Short article → single pass (MAP model with longer output)
        if total_tokens <= self.max_chunk_tokens:
            summary = self._generate_single(text)
            return {
                "summary": summary,
                "chunk_summaries": [summary],
                "num_chunks": 1,
                "method": "single_pass",
                "input_tokens": total_tokens,
            }
        
        # Long article → MapReduce
        return self._map_reduce(text, total_tokens, verbose)
    
    def _map_reduce(self, text: str, total_tokens: int, verbose: bool = False) -> dict:
        """
        MAP:    chunk → summarize each chunk
        REDUCE: concatenate chunk summaries → summarize for coherence
        """
        # === MAP PHASE ===
        chunks = self.chunker.chunk_article(text)
        
        if verbose:
            print(f"  Chunked into {len(chunks)} pieces")
        
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            summary = self._generate_map(chunk.text)
            chunk_summaries.append(summary)
            if verbose:
                print(f"    Chunk {i+1}/{len(chunks)}: {chunk.bpe_token_count} tokens → {len(summary.split())} words")
        
        # === REDUCE PHASE ===
        if len(chunk_summaries) == 1:
            return {
                "summary": chunk_summaries[0],
                "chunk_summaries": chunk_summaries,
                "num_chunks": len(chunks),
                "method": "single_chunk",
                "input_tokens": total_tokens,
            }
        
        # Concatenate all chunk summaries
        merged = " ".join(chunk_summaries)
        merged = fix_bangla_for_tokenizer(merged)
        merged_tokens = count_bpe_tokens(merged)
        
        if verbose:
            print(f"  Merged chunk summaries: {merged_tokens} tokens")
        
        reduce_passes = 0
        
        # If merged summaries still too long, recursively reduce
        while merged_tokens > self.max_chunk_tokens:
            reduce_passes += 1
            if verbose:
                print(f"  Reduce pass {reduce_passes}: {merged_tokens} tokens → re-chunking...")
            
            sub_chunks = self.chunker.chunk_article(merged)
            sub_summaries = []
            for sc in sub_chunks:
                sub_summaries.append(self._generate_reduce(sc.text))
            
            merged = " ".join(sub_summaries)
            merged = fix_bangla_for_tokenizer(merged)
            merged_tokens = count_bpe_tokens(merged)
        
        # Final coherence pass — use REDUCE model
        final_summary = self._generate_reduce(merged)
        
        # Deduplicate sentences
        final_summary = self._dedup_sentences(final_summary)
        
        return {
            "summary": final_summary,
            "chunk_summaries": chunk_summaries,
            "num_chunks": len(chunks),
            "method": "map_reduce",
            "reduce_passes": reduce_passes + 1,
            "input_tokens": total_tokens,
        }
    
    def _generate_single(self, text: str) -> str:
        """Generate summary for short articles (single pass with longer output)."""
        input_text = self.map_prefix + text
        
        inputs = self.map_tokenizer(
            input_text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.map_model.generate(**inputs, **self.single_gen_kwargs)
        summary = self.map_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.strip()
    
    def _generate_map(self, text: str) -> str:
        """Generate summary for a single chunk (MAP phase — Step 1 model)."""
        input_text = self.map_prefix + text
        
        inputs = self.map_tokenizer(
            input_text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.map_model.generate(**inputs, **self.map_gen_kwargs)
        summary = self.map_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.strip()
    
    def _generate_reduce(self, text: str) -> str:
        """Merge chunk summaries into final summary (REDUCE phase — Step 2 model)."""
        input_text = self.reduce_prefix + text
        
        inputs = self.reduce_tokenizer(
            input_text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.reduce_model.generate(**inputs, **self.reduce_gen_kwargs)
        summary = self.reduce_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.strip()
    
    def _dedup_sentences(self, text: str) -> str:
        """Remove duplicate/near-duplicate sentences."""
        sentences = [s.strip() for s in text.split('।') if s.strip()]
        if len(sentences) <= 1:
            return text
        
        unique = []
        for sent in sentences:
            words = set(sent.split())
            is_dup = False
            for existing in unique:
                existing_words = set(existing.split())
                if not words or not existing_words:
                    continue
                intersection = len(words & existing_words)
                union = len(words | existing_words)
                if union > 0 and intersection / union > 0.75:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(sent)
        
        result = '।'.join(unique)
        if text.rstrip().endswith('।'):
            result += '।'
        return result


# ============================================================================
# Test / Demo
# ============================================================================

def run_test(summarizer, test_file="data_splits/test.json", num_samples=10, output_file="pipeline_output.txt"):
    """Test pipeline on long articles from test set."""
    
    print(f"\nLoading test data from {test_file}...")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Filter for LONG articles (>1000 tokens) to test the chunking
    long_articles = []
    for item in test_data:
        tokens = count_bpe_tokens(item['text'])
        if tokens > 1000:
            long_articles.append((item, tokens))
    
    # Also get some short ones for comparison
    short_articles = []
    for item in test_data:
        tokens = count_bpe_tokens(item['text'])
        if tokens <= 1000:
            short_articles.append((item, tokens))
    
    print(f"  Long articles (>1000 tokens): {len(long_articles)}")
    print(f"  Short articles (<=1000 tokens): {len(short_articles)}")
    
    # Test on mix: first few long, then a couple short
    num_long = min(num_samples - 2, len(long_articles))
    num_short = min(2, len(short_articles))
    
    test_items = long_articles[:num_long] + short_articles[:num_short]
    
    print(f"\nTesting on {len(test_items)} articles ({num_long} long + {num_short} short)")
    print("=" * 80)
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write("=" * 80 + "\n")
        f_out.write("CHUNKED SUMMARIZATION PIPELINE OUTPUT\n")
        f_out.write("=" * 80 + "\n\n")
        
        for idx, (item, tokens) in enumerate(test_items):
            text = item['text']
            reference = item['summary']
            
            print(f"\n{'='*80}")
            print(f"Article {idx+1} ({tokens} tokens)")
            print(f"{'='*80}")
            
            result = summarizer.summarize(text, verbose=True)
            
            # Console output
            print(f"\n  Method: {result['method']}")
            print(f"  Chunks: {result['num_chunks']}")
            if 'reduce_passes' in result:
                print(f"  Reduce passes: {result['reduce_passes']}")
            print(f"\n  REFERENCE ({len(reference.split())} words):")
            print(f"  {reference[:200]}...")
            print(f"\n  GENERATED ({len(result['summary'].split())} words):")
            print(f"  {result['summary'][:200]}...")
            
            # File output (full)
            f_out.write(f"{'='*80}\n")
            f_out.write(f"Article {idx+1} | {tokens} tokens | Method: {result['method']} | Chunks: {result['num_chunks']}\n")
            f_out.write(f"{'='*80}\n\n")
            
            f_out.write(f"ARTICLE (first 500 chars):\n{text[:500]}...\n\n")
            f_out.write(f"REFERENCE SUMMARY:\n{reference}\n\n")
            f_out.write(f"GENERATED SUMMARY:\n{result['summary']}\n\n")
            
            if result['num_chunks'] > 1:
                f_out.write(f"CHUNK SUMMARIES:\n")
                for j, cs in enumerate(result['chunk_summaries']):
                    f_out.write(f"  Chunk {j+1}: {cs}\n")
                f_out.write("\n")
            
            f_out.write(f"Stats: {result['num_chunks']} chunks, method={result['method']}\n")
            f_out.write(f"Reference words: {len(reference.split())}, Generated words: {len(result['summary'].split())}\n\n")
    
    print(f"\n{'='*80}")
    print(f"Full output saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunked Summarization Pipeline")
    parser.add_argument('--map_checkpoint', type=str, 
                        default='./banglaT5_full_doc_20260215_123349/checkpoint-7000',
                        help='MAP model checkpoint (Step 1 — chunk summarization)')
    parser.add_argument('--reduce_checkpoint', type=str,
                        default='./banglaT5_reduce_task_20260217_111025/checkpoint-6000',
                        help='REDUCE model checkpoint (Step 2 — merge summaries)')
    parser.add_argument('--text', type=str, default=None, help='Text to summarize')
    parser.add_argument('--file', type=str, default=None, help='File to summarize')
    parser.add_argument('--test', action='store_true', default=True, help='Run test on test set')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of test samples')
    
    args = parser.parse_args()
    
    summarizer = ChunkedSummarizer(
        map_model_path=args.map_checkpoint,
        reduce_model_path=args.reduce_checkpoint,
    )
    
    if args.text:
        result = summarizer.summarize(args.text, verbose=True)
        print(f"\nSummary: {result['summary']}")
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
        result = summarizer.summarize(text, verbose=True)
        print(f"\nSummary: {result['summary']}")
    else:
        run_test(summarizer, num_samples=args.num_samples)
