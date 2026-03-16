"""
Evaluate the chunked summarization pipeline on the FULL test set.
Computes ROUGE-1, ROUGE-2, ROUGE-L (word-level), BERTScore, and Semantic Similarity.
"""

import os
import json
import time
import numpy as np
import torch
from tqdm import tqdm
from rouge_score import rouge_scorer

from run_pipeline import ChunkedSummarizer, fix_bangla_for_tokenizer

# BERTScore
try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("⚠️  BERTScore not available")
    BERTSCORE_AVAILABLE = False

# Semantic Similarity
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    SEMANTIC_AVAILABLE = True
except ImportError:
    print("⚠️  sentence-transformers not available")
    SEMANTIC_AVAILABLE = False


class WordTokenizer:
    def tokenize(self, text):
        return text.split()


def evaluate_pipeline(
    map_checkpoint="./banglaT5_full_doc_20260215_123349/checkpoint-7000",
    reduce_checkpoint="./banglaT5_reduce_task_20260217_111025/checkpoint-6000",
    test_file="data_splits/test.json",
    output_file="pipeline_eval_results.json",
    batch_size_bert=64,
):
    # Load test data
    print(f"Loading test data from {test_file}...")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"  Total test samples: {len(test_data)}")

    # Initialize pipeline with 2 models
    summarizer = ChunkedSummarizer(
        map_model_path=map_checkpoint,
        reduce_model_path=reduce_checkpoint,
    )

    # Generate all summaries (with periodic checkpointing)
    CHECKPOINT_FILE = "eval_pipeline_checkpoint.json"
    print(f"\nGenerating summaries for {len(test_data)} articles...")
    predictions = []
    references = []
    methods = {"single_pass": 0, "single_chunk": 0, "map_reduce": 0}
    chunk_counts = []
    start_idx = 0
    
    # Resume from checkpoint if exists
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            ckpt = json.load(f)
        predictions = ckpt['predictions']
        references = ckpt['references']
        methods = ckpt['methods']
        chunk_counts = ckpt['chunk_counts']
        start_idx = ckpt['next_idx']
        print(f"  Resuming from checkpoint: {start_idx}/{len(test_data)} already done")
    
    start_time = time.time()
    
    for i, item in enumerate(tqdm(test_data[start_idx:], desc="Summarizing", initial=start_idx, total=len(test_data))):
        result = summarizer.summarize(item['text'])
        predictions.append(result['summary'])
        references.append(fix_bangla_for_tokenizer(item['summary']))
        methods[result['method']] = methods.get(result['method'], 0) + 1
        chunk_counts.append(result['num_chunks'])
        
        # Save checkpoint every 100 articles
        if (start_idx + i + 1) % 100 == 0:
            with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    'predictions': predictions,
                    'references': references,
                    'methods': methods,
                    'chunk_counts': chunk_counts,
                    'next_idx': start_idx + i + 1,
                }, f, ensure_ascii=False)
            tqdm.write(f"  [Checkpoint saved: {start_idx + i + 1}/{len(test_data)}]")
    
    gen_time = time.time() - start_time
    print(f"\nGeneration done in {gen_time:.1f}s ({gen_time/len(test_data):.2f}s per article)")
    
    # Method distribution
    print(f"\nMethod distribution:")
    for method, count in methods.items():
        print(f"  {method}: {count} ({count/len(test_data)*100:.1f}%)")
    print(f"  Average chunks per article: {np.mean(chunk_counts):.2f}")
    print(f"  Max chunks: {max(chunk_counts)}")

    # === ROUGE ===
    print("\nComputing ROUGE scores (word-level)...")
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], 
        use_stemmer=False, 
        tokenizer=WordTokenizer()
    )
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
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

    # === BERTScore ===
    avg_bertscore = None
    if BERTSCORE_AVAILABLE:
        print("\nComputing BERTScore...")
        try:
            P, R, F1 = bert_score_fn(
                predictions, references, 
                lang='bn', verbose=True,
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=batch_size_bert
            )
            avg_bertscore = F1.mean().item()
            print(f"  BERTScore F1: {avg_bertscore:.4f}")
        except Exception as e:
            print(f"  BERTScore failed: {e}")

    # === Semantic Similarity ===
    avg_semantic = None
    if SEMANTIC_AVAILABLE:
        print("\nComputing Semantic Similarity...")
        try:
            sem_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
            sem_model.to("cuda" if torch.cuda.is_available() else "cpu")
            
            ref_embs = sem_model.encode(references, convert_to_tensor=True, show_progress_bar=True, batch_size=64)
            pred_embs = sem_model.encode(predictions, convert_to_tensor=True, show_progress_bar=True, batch_size=64)
            
            cos_scores = st_util.cos_sim(ref_embs, pred_embs).diagonal().cpu().numpy()
            avg_semantic = float(np.mean(cos_scores))
            print(f"  Semantic Similarity: {avg_semantic:.4f}")
        except Exception as e:
            print(f"  Semantic similarity failed: {e}")

    # === Length stats ===
    pred_lengths = [len(p.split()) for p in predictions]
    ref_lengths = [len(r.split()) for r in references]
    
    # === Summary ===
    results = {
        "test_samples": len(test_data),
        "generation_time_seconds": round(gen_time, 1),
        "seconds_per_article": round(gen_time / len(test_data), 2),
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
        "checkpoint_map": map_checkpoint,
        "checkpoint_reduce": reduce_checkpoint,
    }
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("PIPELINE EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"  Test samples:       {results['test_samples']}")
    print(f"  Time:               {results['generation_time_seconds']}s ({results['seconds_per_article']}s/article)")
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
    print(f"{'='*80}")
    print(f"  Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    evaluate_pipeline()
