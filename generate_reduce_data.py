"""
Generate Reduce-Task Training Data (Paper-Quality)

Uses the trained Step 1 teacher (checkpoint-7000) to generate REAL chunk
summaries for every article — matching the inference-time distribution.

Core idea:
  For each article:
    1. Chunk it the same way the pipeline does (sentence-aligned, overlapping)
    2. Generate chunk summaries using the teacher model
    3. Concatenate chunk summaries → input
    4. Gold summary → target

Augmentation for robustness (publishable trick):
  - Shuffle chunk order  (teaches order-invariance)
  - Drop one chunk       (teaches missing-info tolerance)
  - Duplicate a sentence (teaches deduplication)
  - No augmentation      (clean baseline)

Output: JSON file ready for train_reduce_task.py

Usage:
    python generate_reduce_data.py
    python generate_reduce_data.py --split train --batch_size 8
    python generate_reduce_data.py --split val
    python generate_reduce_data.py --split test
"""

import os
import sys
import json
import random
import argparse
import unicodedata
import torch
from typing import List, Dict
from tqdm import tqdm

from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
from bangla_sentence_splitter import split_sentences, count_bpe_tokens
from chunk_processor import SentenceChunker

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def fix_bangla_for_tokenizer(text):
    """Decompose precomposed Bangla chars that cause <unk> in BanglaT5."""
    text = text.replace('\u09DF', '\u09AF\u09BC')
    text = text.replace('\u09DC', '\u09A1\u09BC')
    text = text.replace('\u09DD', '\u09A2\u09BC')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201C', '"').replace('\u201D', '"')
    text = text.replace('\u2014', '-').replace('\u2013', '-')
    text = text.replace('\u2026', '...')
    text = text.replace('\u200c', '').replace('\u200d', '')
    return text


def normalize_bangla(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace('\u200d', '').replace('\u200c', '')
    text = ' '.join(text.split())
    return text.strip()


# ============================================================================
# TEACHER MODEL
# ============================================================================

class TeacherSummarizer:
    """Wraps checkpoint-7000 for generating chunk summaries."""
    
    def __init__(self, model_path: str, device: str = None, batch_size: int = 8):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        print(f"Loading teacher model from {model_path}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"  Teacher loaded on {self.device}")
        
        self.chunker = SentenceChunker(max_tokens=900, overlap_sentences=3)
        self.input_prefix = "summarize bangla news: "
        
        # Generation config — deterministic beam search
        # Natural noise from model imperfection is enough
        self.gen_kwargs = {
            "max_length": 256,
            "min_length": 30,
            "num_beams": 4,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.0,
            "early_stopping": True,
        }
    
    def chunk_article(self, text: str) -> List[str]:
        """
        Chunk an article into text pieces. Returns list of chunk texts.
        Short articles return as single chunk.
        """
        text = fix_bangla_for_tokenizer(text)
        total_tokens = count_bpe_tokens(text)
        
        if total_tokens <= 900:
            return [text]
        
        chunks = self.chunker.chunk_article(text)
        return [chunk.text for chunk in chunks]
    
    @torch.no_grad()
    def generate_batch(self, texts: List[str]) -> List[str]:
        """
        Generate summaries for a batch of texts at once.
        This maximizes GPU utilization vs one-by-one generation.
        """
        if not texts:
            return []
        
        input_texts = [self.input_prefix + t for t in texts]
        
        inputs = self.tokenizer(
            input_texts,
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.generate(**inputs, **self.gen_kwargs)
        summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [s.strip() for s in summaries]


# ============================================================================
# CORRUPTION AUGMENTATION
# ============================================================================

def augment_shuffle(chunk_summaries: List[str]) -> List[str]:
    """Shuffle chunk order — teaches order-invariance."""
    if len(chunk_summaries) <= 1:
        return chunk_summaries
    shuffled = chunk_summaries.copy()
    random.shuffle(shuffled)
    return shuffled


def augment_drop(chunk_summaries: List[str]) -> List[str]:
    """Drop one random chunk — teaches missing-info tolerance."""
    if len(chunk_summaries) <= 1:
        return chunk_summaries
    idx = random.randint(0, len(chunk_summaries) - 1)
    return [s for i, s in enumerate(chunk_summaries) if i != idx]


def augment_duplicate(chunk_summaries: List[str]) -> List[str]:
    """Duplicate one random sentence within a chunk — teaches deduplication."""
    if not chunk_summaries:
        return chunk_summaries
    
    augmented = chunk_summaries.copy()
    idx = random.randint(0, len(augmented) - 1)
    
    sentences = split_sentences(augmented[idx])
    if len(sentences) >= 2:
        dup_idx = random.randint(0, len(sentences) - 1)
        sentences.insert(dup_idx + 1, sentences[dup_idx])
        augmented[idx] = ' '.join(sentences)
    
    return augmented


AUGMENTATION_FNS = {
    "clean": lambda x: x,              # No augmentation (baseline)
    "shuffle": augment_shuffle,         # Shuffle chunk order
    "drop": augment_drop,              # Drop one chunk
    "duplicate": augment_duplicate,     # Duplicate a sentence
}


# ============================================================================
# DATA GENERATION
# ============================================================================

def create_reduce_samples(
    article_text: str,
    gold_summary: str,
    chunk_summaries: List[str],
    augment: bool = True,
) -> List[Dict]:
    """
    Create reduce training samples from teacher-generated chunk summaries.
    
    Returns:
        List of dicts with 'text' (concatenated chunk summaries) and 'summary' (gold)
        Multiple samples per article if augmentation is enabled.
    """
    samples = []
    
    # 1. CLEAN sample (always included)
    concat = " [CHUNK] ".join(chunk_summaries)
    samples.append({
        "text": concat,
        "summary": gold_summary,
        "augmentation": "clean",
        "num_chunks": len(chunk_summaries),
    })
    
    if not augment or len(chunk_summaries) <= 1:
        return samples
    
    # 2. SHUFFLE sample (only for multi-chunk)
    shuffled = augment_shuffle(chunk_summaries)
    concat_shuffled = " [CHUNK] ".join(shuffled)
    if concat_shuffled != concat:  # Only add if actually different
        samples.append({
            "text": concat_shuffled,
            "summary": gold_summary,
            "augmentation": "shuffle",
            "num_chunks": len(shuffled),
        })
    
    # 3. DROP sample (only for 3+ chunks — dropping from 2 leaves 1, too easy)
    if len(chunk_summaries) >= 3:
        dropped = augment_drop(chunk_summaries)
        concat_dropped = " [CHUNK] ".join(dropped)
        samples.append({
            "text": concat_dropped,
            "summary": gold_summary,
            "augmentation": "drop",
            "num_chunks": len(dropped),
        })
    
    # 4. DUPLICATE sample
    duplicated = augment_duplicate(chunk_summaries)
    concat_dup = " [CHUNK] ".join(duplicated)
    if concat_dup != concat:
        samples.append({
            "text": concat_dup,
            "summary": gold_summary,
            "augmentation": "duplicate",
            "num_chunks": len(duplicated),
        })
    
    return samples


def process_split(
    data: List[Dict],
    teacher: TeacherSummarizer,
    split_name: str,
    augment: bool = True,
    gen_batch_size: int = 16,
    output_dir: str = "reduce_data",
) -> List[Dict]:
    """Process an entire data split with BATCHED generation and CHECKPOINT RESUME."""
    
    print(f"\nProcessing {split_name} split: {len(data)} articles")
    print(f"  Augmentation: {'ON' if augment else 'OFF'}")
    print(f"  Generation batch size: {gen_batch_size}")
    
    # ── PHASE 1: Filter & chunk all articles (CPU, fast) ──
    print(f"  Phase 1: Filtering & chunking...")
    valid_articles = []  # (text, summary, chunk_texts)
    skipped = 0
    
    for item in tqdm(data, desc=f"Chunking {split_name}"):
        text = str(item.get("text", "")).strip()
        summary = str(item.get("summary", "")).strip()
        
        text = normalize_bangla(text)
        summary = normalize_bangla(summary)
        
        # Quality filters
        if len(text) < 100 or len(summary) < 20:
            skipped += 1
            continue
        
        text_words = text.split()
        summary_words = summary.split()
        
        if len(text_words) < 30 or len(summary_words) < 8:
            skipped += 1
            continue
        
        if len(summary_words) > 150:
            skipped += 1
            continue
        
        try:
            chunk_texts = teacher.chunk_article(text)
        except Exception as e:
            print(f"\n  Warning: Chunking failed ({len(text)} chars): {e}")
            skipped += 1
            continue
        
        if not chunk_texts:
            skipped += 1
            continue
        
        valid_articles.append((text, summary, chunk_texts))
    
    print(f"  Filtered: {len(valid_articles)} valid, {skipped} skipped")
    
    # ── PHASE 2: Flatten all chunks & batch-generate (GPU) with checkpointing ──
    all_chunks = []
    chunk_mapping = []
    
    for art_idx, (text, summary, chunk_texts) in enumerate(valid_articles):
        for c_idx, ct in enumerate(chunk_texts):
            all_chunks.append(ct)
            chunk_mapping.append((art_idx, c_idx))
    
    total_chunks = len(all_chunks)
    checkpoint_path = os.path.join(output_dir, f".checkpoint_{split_name}.json")
    
    # Check for existing checkpoint (resume support)
    start_idx = 0
    all_summaries = []
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                ckpt = json.load(f)
            all_summaries = ckpt.get("summaries", [])
            start_idx = len(all_summaries)
            if start_idx > 0 and start_idx < total_chunks:
                print(f"  ⚡ RESUMING from checkpoint: {start_idx}/{total_chunks} chunks already done")
            elif start_idx >= total_chunks:
                print(f"  ⚡ All {total_chunks} chunks already generated, skipping to Phase 3")
        except Exception as e:
            print(f"  Warning: Checkpoint corrupted, starting fresh: {e}")
            start_idx = 0
            all_summaries = []
    
    if start_idx < total_chunks:
        print(f"  Phase 2: Generating summaries for {total_chunks - start_idx} chunks "
              f"(of {total_chunks} total) in batches of {gen_batch_size}...")
        
        save_every = 500  # Save checkpoint every N batches
        batch_count = 0
        
        for batch_start in tqdm(
            range(start_idx, total_chunks, gen_batch_size),
            desc=f"Batch-gen {split_name}",
            initial=start_idx // gen_batch_size,
            total=(total_chunks + gen_batch_size - 1) // gen_batch_size
        ):
            batch = all_chunks[batch_start:batch_start + gen_batch_size]
            try:
                batch_summaries = teacher.generate_batch(batch)
                all_summaries.extend(batch_summaries)
            except Exception as e:
                print(f"\n  Warning: Batch generation failed at {batch_start}: {e}")
                for t in batch:
                    try:
                        s = teacher.generate_batch([t])[0]
                        all_summaries.append(s)
                    except:
                        all_summaries.append("")
            
            batch_count += 1
            
            # Save checkpoint periodically
            if batch_count % save_every == 0:
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump({"summaries": all_summaries}, f, ensure_ascii=False)
                tqdm.write(f"  💾 Checkpoint saved: {len(all_summaries)}/{total_chunks} chunks")
        
        # Final checkpoint save
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump({"summaries": all_summaries}, f, ensure_ascii=False)
    
    # ── PHASE 3: Reassemble chunk summaries per article ──
    print(f"  Phase 3: Assembling reduce samples...")
    
    article_chunk_summaries = [[] for _ in valid_articles]
    for i, (art_idx, c_idx) in enumerate(chunk_mapping):
        if i < len(all_summaries):
            article_chunk_summaries[art_idx].append(all_summaries[i])
    
    reduce_samples = []
    multi_chunk = 0
    
    for art_idx, (text, summary, chunk_texts) in enumerate(valid_articles):
        chunk_summaries = article_chunk_summaries[art_idx]
        
        if not chunk_summaries or all(not s.strip() for s in chunk_summaries):
            continue
        
        if len(chunk_summaries) > 1:
            multi_chunk += 1
        
        use_augment = augment and split_name == "train"
        samples = create_reduce_samples(text, summary, chunk_summaries, augment=use_augment)
        reduce_samples.extend(samples)
    
    # Clean up checkpoint after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"  🗑️ Checkpoint cleaned up")
    
    print(f"\n  {split_name} results:")
    print(f"    Articles processed: {len(data) - skipped}")
    print(f"    Articles skipped: {skipped}")
    print(f"    Multi-chunk articles: {multi_chunk}")
    print(f"    Total reduce samples: {len(reduce_samples)}")
    
    # Show augmentation breakdown
    aug_counts = {}
    for s in reduce_samples:
        aug = s.get("augmentation", "clean")
        aug_counts[aug] = aug_counts.get(aug, 0) + 1
    print(f"    Augmentation breakdown: {aug_counts}")
    
    return reduce_samples


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate Reduce Task Training Data')
    parser.add_argument('--model', type=str,
                       default='./banglaT5_full_doc_20260215_123349/checkpoint-7000',
                       help='Teacher model checkpoint')
    parser.add_argument('--train_file', type=str, default='data_splits/train.json')
    parser.add_argument('--val_file', type=str, default='data_splits/val.json')
    parser.add_argument('--test_file', type=str, default='data_splits/test.json')
    parser.add_argument('--output_dir', type=str, default='reduce_data',
                       help='Output directory for generated data')
    parser.add_argument('--split', type=str, default='all',
                       choices=['all', 'train', 'val', 'test'],
                       help='Which split to process (default: all)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for GPU generation (higher = faster, more VRAM)')
    parser.add_argument('--no_augment', action='store_true',
                       help='Disable corruption augmentation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("=" * 80)
    print("REDUCE DATA GENERATION — Teacher Chunk Summaries")
    print("=" * 80)
    print(f"Teacher model: {args.model}")
    print(f"Output dir: {args.output_dir}")
    print(f"Generation batch size: {args.batch_size}")
    print(f"Augmentation: {'OFF' if args.no_augment else 'ON (shuffle + drop + duplicate)'}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load teacher
    teacher = TeacherSummarizer(args.model, batch_size=args.batch_size)
    
    splits_to_process = []
    if args.split in ('all', 'train'):
        splits_to_process.append(('train', args.train_file))
    if args.split in ('all', 'val'):
        splits_to_process.append(('val', args.val_file))
    if args.split in ('all', 'test'):
        splits_to_process.append(('test', args.test_file))
    
    for split_name, split_file in splits_to_process:
        print(f"\n{'='*80}")
        print(f"PROCESSING: {split_name}")
        print(f"{'='*80}")
        
        with open(split_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        reduce_samples = process_split(
            data, teacher, split_name,
            augment=not args.no_augment,
            gen_batch_size=args.batch_size,
            output_dir=args.output_dir
        )
        
        # Save
        output_path = os.path.join(args.output_dir, f"reduce_{split_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(reduce_samples, f, ensure_ascii=False, indent=1)
        
        print(f"  Saved {len(reduce_samples)} samples → {output_path}")
    
    print(f"\n{'='*80}")
    print("DONE — Reduce data generated")
    print(f"{'='*80}")
    print(f"\nNext step: python train_reduce_task.py --data_dir {args.output_dir}")


if __name__ == "__main__":
    main()
