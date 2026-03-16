"""
Batched GPU Evaluation — MapReduce Summarization Pipeline
Processes all articles in parallel batches for maximum GPU utilization.

Strategy:
  1. Pre-chunk ALL articles (CPU)
  2. Batch MAP: Generate all chunk summaries in GPU batches
  3. Assemble REDUCE inputs per article
  4. Batch REDUCE: Generate all final summaries in GPU batches
  5. Compute metrics (ROUGE, BERTScore, Semantic Similarity)
"""

import os
import json
import time
import math
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple
from rouge_score import rouge_scorer
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

from bangla_sentence_splitter import count_bpe_tokens
from chunk_processor import SentenceChunker
from run_pipeline import fix_bangla_for_tokenizer

# Optional metrics
try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("⚠️  BERTScore not available")
    BERTSCORE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    SEMANTIC_AVAILABLE = True
except ImportError:
    print("⚠️  sentence-transformers not available")
    SEMANTIC_AVAILABLE = False


# ── Generation configs ──────────────────────────────────────────────────────
MAP_GEN_KWARGS = {
    "max_length": 256,
    "min_length": 40,
    "num_beams": 5,
    "length_penalty": 1.2,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
}

REDUCE_GEN_KWARGS = {
    "max_length": 256,
    "min_length": 80,
    "num_beams": 5,
    "length_penalty": 2.0,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
}

SINGLE_GEN_KWARGS = {
    "max_length": 256,
    "min_length": 60,
    "num_beams": 5,
    "length_penalty": 1.5,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
}


def dedup_sentences(text: str) -> str:
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


@torch.no_grad()
def batch_generate(
    model, tokenizer, texts: List[str], gen_kwargs: dict,
    batch_size: int = 16, device: str = "cuda", desc: str = "Generating"
) -> List[str]:
    """Generate summaries for a list of texts in GPU batches."""
    all_summaries = []
    num_batches = math.ceil(len(texts) / batch_size)
    
    for batch_idx in tqdm(range(num_batches), desc=desc, leave=False):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        
        # Tokenize batch with padding
        inputs = tokenizer(
            batch_texts,
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            **gen_kwargs
        )
        
        # Decode
        summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_summaries.extend([s.strip() for s in summaries])
    
    return all_summaries


def evaluate_batched(
    map_checkpoint="./banglaT5_full_doc_20260215_123349/checkpoint-7000",
    reduce_checkpoint="./banglaT5_reduce_task_20260217_111025/checkpoint-6000",
    test_file="data_splits/test.json",
    output_file="pipeline_eval_results.json",
    map_batch_size=16,
    reduce_batch_size=16,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # ── Load test data ────────────────────────────────────────────
    print(f"\nLoading test data from {test_file}...")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"  Total test samples: {len(test_data)}")
    
    # ── Load models ───────────────────────────────────────────────
    print(f"\nLoading MAP model from {map_checkpoint}...")
    map_tokenizer = T5Tokenizer.from_pretrained(map_checkpoint, use_fast=False)
    map_model = AutoModelForSeq2SeqLM.from_pretrained(
        map_checkpoint, torch_dtype=torch.bfloat16
    ).to(device).eval()
    print(f"  MAP model loaded on {device} (bf16)")
    
    print(f"Loading REDUCE model from {reduce_checkpoint}...")
    reduce_tokenizer = T5Tokenizer.from_pretrained(reduce_checkpoint, use_fast=False)
    reduce_model = AutoModelForSeq2SeqLM.from_pretrained(
        reduce_checkpoint, torch_dtype=torch.bfloat16
    ).to(device).eval()
    print(f"  REDUCE model loaded on {device} (bf16)")
    
    # ── Phase 1: Pre-chunk all articles (CPU) ────────────────────
    print(f"\n{'='*60}")
    print("PHASE 1: Chunking all articles...")
    print(f"{'='*60}")
    
    chunker = SentenceChunker(max_tokens=900, overlap_sentences=3)
    map_prefix = "summarize bangla news: "
    reduce_prefix = "summarize multiple summaries: "
    
    # Classify articles into single-pass vs multi-chunk
    single_pass_indices = []      # articles < 900 tokens
    single_pass_texts = []
    
    multi_chunk_indices = []      # articles needing MAP+REDUCE
    multi_chunk_data = []         # list of (article_idx, [chunk_texts])
    
    all_map_texts = []            # flat list of ALL chunk texts (for batching)
    map_text_to_article = []      # maps each map_text index → (article_group_idx, chunk_idx_within_article)
    
    references = []
    
    for idx, item in enumerate(tqdm(test_data, desc="Chunking")):
        text = fix_bangla_for_tokenizer(item['text'])
        ref = fix_bangla_for_tokenizer(item['summary'])
        references.append(ref)
        
        total_tokens = count_bpe_tokens(text)
        
        if total_tokens <= 900:
            # Single pass — use MAP model directly
            single_pass_indices.append(idx)
            single_pass_texts.append(map_prefix + text)
        else:
            # Multi-chunk → need MAP then REDUCE
            chunks = chunker.chunk_article(text)
            chunk_texts = [map_prefix + c.text for c in chunks]
            
            group_idx = len(multi_chunk_data)
            multi_chunk_indices.append(idx)
            multi_chunk_data.append({
                'article_idx': idx,
                'num_chunks': len(chunks),
                'map_start': len(all_map_texts),  # where this article's chunks start in flat list
            })
            
            all_map_texts.extend(chunk_texts)
    
    print(f"\n  Single-pass articles: {len(single_pass_indices)}")
    print(f"  Multi-chunk articles: {len(multi_chunk_indices)}")
    print(f"  Total MAP chunks to process: {len(all_map_texts)}")
    avg_chunks = len(all_map_texts) / max(len(multi_chunk_indices), 1)
    print(f"  Average chunks per multi-chunk article: {avg_chunks:.1f}")
    
    # ── Phase 2: Batch MAP generation ────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 2: Batch MAP generation ({len(single_pass_texts) + len(all_map_texts)} total inputs)")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # 2a: Generate single-pass summaries
    single_summaries = []
    if single_pass_texts:
        print(f"\n  Generating {len(single_pass_texts)} single-pass summaries (batch_size={map_batch_size})...")
        single_summaries = batch_generate(
            map_model, map_tokenizer, single_pass_texts, SINGLE_GEN_KWARGS,
            batch_size=map_batch_size, device=device, desc="Single-pass MAP"
        )
    
    # 2b: Generate all chunk summaries for multi-chunk articles
    chunk_summaries_flat = []
    if all_map_texts:
        print(f"\n  Generating {len(all_map_texts)} chunk summaries (batch_size={map_batch_size})...")
        chunk_summaries_flat = batch_generate(
            map_model, map_tokenizer, all_map_texts, MAP_GEN_KWARGS,
            batch_size=map_batch_size, device=device, desc="Multi-chunk MAP"
        )
    
    map_time = time.time() - start_time
    print(f"\n  MAP phase done in {map_time:.1f}s")
    
    # ── Free MAP model memory for REDUCE ─────────────────────────
    del map_model
    torch.cuda.empty_cache()
    print("  MAP model freed from GPU")
    
    # ── Phase 3: Assemble REDUCE inputs ──────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 3: Assembling REDUCE inputs...")
    print(f"{'='*60}")
    
    reduce_texts = []        # texts for REDUCE model
    reduce_article_indices = []  # which article each reduce input belongs to
    needs_recursive = []     # articles needing recursive reduce
    
    for group in multi_chunk_data:
        article_idx = group['article_idx']
        start = group['map_start']
        num_chunks = group['num_chunks']
        
        # Get this article's chunk summaries
        article_chunk_sums = chunk_summaries_flat[start:start + num_chunks]
        
        if num_chunks == 1:
            # Single chunk — no reduce needed, use chunk summary directly
            reduce_texts.append(None)  # placeholder
            reduce_article_indices.append(article_idx)
            # Store the summary directly
            group['final_summary'] = article_chunk_sums[0]
            group['method'] = 'single_chunk'
        else:
            # Merge chunk summaries
            merged = " ".join(article_chunk_sums)
            merged = fix_bangla_for_tokenizer(merged)
            merged_tokens = count_bpe_tokens(merged)
            
            if merged_tokens <= 900:
                # Fits in one REDUCE pass
                reduce_texts.append(reduce_prefix + merged)
                reduce_article_indices.append(article_idx)
                group['method'] = 'map_reduce'
                group['final_summary'] = None  # will be filled after REDUCE
                group['reduce_idx'] = len(reduce_texts) - 1
            else:
                # Needs recursive reduction — handle separately
                needs_recursive.append((group, merged))
                group['method'] = 'map_reduce'
                group['final_summary'] = None
                group['reduce_idx'] = None
    
    # Filter out None placeholders (single_chunk articles)
    actual_reduce_texts = [t for t in reduce_texts if t is not None]
    reduce_idx_map = {}  # old index → new index in actual_reduce_texts
    new_idx = 0
    for old_idx, t in enumerate(reduce_texts):
        if t is not None:
            reduce_idx_map[old_idx] = new_idx
            new_idx += 1
    
    print(f"  Direct REDUCE inputs: {len(actual_reduce_texts)}")
    print(f"  Recursive REDUCE needed: {len(needs_recursive)}")
    
    # ── Phase 4: Batch REDUCE generation ─────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 4: Batch REDUCE generation ({len(actual_reduce_texts)} inputs)")
    print(f"{'='*60}")
    
    reduce_start = time.time()
    
    reduce_summaries = []
    if actual_reduce_texts:
        reduce_summaries = batch_generate(
            reduce_model, reduce_tokenizer, actual_reduce_texts, REDUCE_GEN_KWARGS,
            batch_size=reduce_batch_size, device=device, desc="REDUCE"
        )
    
    # Fill in results for direct reduce articles
    for group in multi_chunk_data:
        if group['final_summary'] is not None:
            continue  # single_chunk, already has summary
        if group.get('reduce_idx') is not None and group['reduce_idx'] in reduce_idx_map:
            new_idx = reduce_idx_map[group['reduce_idx']]
            summary = reduce_summaries[new_idx]
            group['final_summary'] = dedup_sentences(summary)
    
    # ── Phase 4b: Handle recursive reduce (few articles) ─────────
    if needs_recursive:
        print(f"\n  Processing {len(needs_recursive)} recursive REDUCE articles...")
        for group, merged in tqdm(needs_recursive, desc="Recursive REDUCE"):
            merged_tokens = count_bpe_tokens(merged)
            passes = 0
            while merged_tokens > 900:
                passes += 1
                sub_chunks = chunker.chunk_article(merged)
                sub_texts = [reduce_prefix + sc.text for sc in sub_chunks]
                sub_sums = batch_generate(
                    reduce_model, reduce_tokenizer, sub_texts, REDUCE_GEN_KWARGS,
                    batch_size=reduce_batch_size, device=device,
                    desc=f"Recursive pass {passes}"
                )
                merged = " ".join(sub_sums)
                merged = fix_bangla_for_tokenizer(merged)
                merged_tokens = count_bpe_tokens(merged)
            
            # Final reduce
            final_input = [reduce_prefix + merged]
            final_sum = batch_generate(
                reduce_model, reduce_tokenizer, final_input, REDUCE_GEN_KWARGS,
                batch_size=1, device=device, desc="Final reduce"
            )
            group['final_summary'] = dedup_sentences(final_sum[0])
    
    reduce_time = time.time() - reduce_start
    total_time = time.time() - start_time
    print(f"\n  REDUCE phase done in {reduce_time:.1f}s")
    print(f"  Total generation time: {total_time:.1f}s")
    
    # ── Phase 5: Assemble all predictions ────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 5: Assembling predictions & computing metrics...")
    print(f"{'='*60}")
    
    predictions = [""] * len(test_data)
    methods = {"single_pass": 0, "single_chunk": 0, "map_reduce": 0}
    chunk_counts = [0] * len(test_data)
    
    # Fill single-pass predictions
    for i, idx in enumerate(single_pass_indices):
        predictions[idx] = single_summaries[i]
        methods['single_pass'] += 1
        chunk_counts[idx] = 1
    
    # Fill multi-chunk predictions
    for group in multi_chunk_data:
        idx = group['article_idx']
        predictions[idx] = group['final_summary'] or ""
        methods[group['method']] = methods.get(group['method'], 0) + 1
        chunk_counts[idx] = group['num_chunks']
    
    # Sanity check
    empty_count = sum(1 for p in predictions if not p)
    if empty_count > 0:
        print(f"  ⚠️  {empty_count} empty predictions!")
    
    # ── ROUGE ────────────────────────────────────────────────────
    print("\nComputing ROUGE scores...")
    
    class WordTokenizer:
        def tokenize(self, text):
            return text.split()
    
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=False,
        tokenizer=WordTokenizer()
    )
    
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    avg_rouge1 = np.mean(rouge1_scores)
    avg_rouge2 = np.mean(rouge2_scores)
    avg_rougeL = np.mean(rougeL_scores)
    
    print(f"  ROUGE-1: {avg_rouge1:.4f}")
    print(f"  ROUGE-2: {avg_rouge2:.4f}")
    print(f"  ROUGE-L: {avg_rougeL:.4f}")
    
    # ── BERTScore ────────────────────────────────────────────────
    avg_bertscore = None
    if BERTSCORE_AVAILABLE:
        print("\nComputing BERTScore...")
        try:
            # Free REDUCE model for BERTScore
            del reduce_model
            torch.cuda.empty_cache()
            
            P, R, F1 = bert_score_fn(
                predictions, references,
                lang='bn', verbose=True,
                device=device, batch_size=64
            )
            avg_bertscore = F1.mean().item()
            print(f"  BERTScore F1: {avg_bertscore:.4f}")
        except Exception as e:
            print(f"  BERTScore failed: {e}")
    
    # ── Semantic Similarity ──────────────────────────────────────
    avg_semantic = None
    if SEMANTIC_AVAILABLE:
        print("\nComputing Semantic Similarity...")
        try:
            torch.cuda.empty_cache()
            sem_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
            sem_model.to(device)
            
            ref_embs = sem_model.encode(references, convert_to_tensor=True, show_progress_bar=True, batch_size=64)
            pred_embs = sem_model.encode(predictions, convert_to_tensor=True, show_progress_bar=True, batch_size=64)
            
            cos_scores = st_util.cos_sim(ref_embs, pred_embs).diagonal().cpu().numpy()
            avg_semantic = float(np.mean(cos_scores))
            print(f"  Semantic Similarity: {avg_semantic:.4f}")
        except Exception as e:
            print(f"  Semantic similarity failed: {e}")
    
    # ── Length stats ──────────────────────────────────────────────
    pred_lengths = [len(p.split()) for p in predictions]
    ref_lengths = [len(r.split()) for r in references]
    
    # ── Final Results ────────────────────────────────────────────
    results = {
        "test_samples": len(test_data),
        "generation_time_seconds": round(total_time, 1),
        "seconds_per_article": round(total_time / len(test_data), 2),
        "map_time_seconds": round(map_time, 1),
        "reduce_time_seconds": round(reduce_time, 1),
        "rouge1": round(avg_rouge1, 4),
        "rouge2": round(avg_rouge2, 4),
        "rougeL": round(avg_rougeL, 4),
        "bertscore_f1": round(avg_bertscore, 4) if avg_bertscore else None,
        "semantic_similarity": round(avg_semantic, 4) if avg_semantic else None,
        "method_distribution": methods,
        "avg_chunks": round(np.mean(chunk_counts), 2),
        "max_chunks": int(max(chunk_counts)),
        "avg_pred_words": round(np.mean(pred_lengths), 1),
        "avg_ref_words": round(np.mean(ref_lengths), 1),
        "map_batch_size": map_batch_size,
        "reduce_batch_size": reduce_batch_size,
        "checkpoint_map": map_checkpoint,
        "checkpoint_reduce": reduce_checkpoint,
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("PIPELINE EVALUATION RESULTS (BATCHED)")
    print(f"{'='*80}")
    print(f"  Test samples:       {results['test_samples']}")
    print(f"  Total time:         {results['generation_time_seconds']}s ({results['seconds_per_article']}s/article)")
    print(f"  MAP time:           {results['map_time_seconds']}s")
    print(f"  REDUCE time:        {results['reduce_time_seconds']}s")
    print(f"  ROUGE-1:            {results['rouge1']}")
    print(f"  ROUGE-2:            {results['rouge2']}")
    print(f"  ROUGE-L:            {results['rougeL']}")
    if results['bertscore_f1']:
        print(f"  BERTScore F1:       {results['bertscore_f1']}")
    if results['semantic_similarity']:
        print(f"  Semantic Sim:       {results['semantic_similarity']}")
    print(f"  Avg pred words:     {results['avg_pred_words']}")
    print(f"  Avg ref words:      {results['avg_ref_words']}")
    print(f"  Method split:       {results['method_distribution']}")
    print(f"  Avg chunks:         {results['avg_chunks']}")
    print(f"  Batch sizes:        MAP={map_batch_size}, REDUCE={reduce_batch_size}")
    print(f"{'='*80}")
    print(f"  Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batched GPU Pipeline Evaluation")
    parser.add_argument('--map_batch_size', type=int, default=16, help='MAP batch size')
    parser.add_argument('--reduce_batch_size', type=int, default=16, help='REDUCE batch size')
    parser.add_argument('--output', type=str, default='pipeline_eval_results.json')
    args = parser.parse_args()
    
    evaluate_batched(
        map_batch_size=args.map_batch_size,
        reduce_batch_size=args.reduce_batch_size,
        output_file=args.output,
    )
