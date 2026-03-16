import json
import torch
from tqdm import tqdm
from run_pipeline import ChunkedSummarizer
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

print("Loading models...")

# Initialize the chunked summarizer with both models
summarizer = ChunkedSummarizer(
    map_model_path="./banglaT5_full_doc_20260215_123349/checkpoint-7000",
    reduce_model_path="./banglaT5_reduce_task_20260217_111025/checkpoint-6000",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Load sentence transformer for semantic similarity
semantic_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

print("\nLoading test data...")
with open('test_remaining_388.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"Loaded {len(test_data)} test samples")

# Storage for results
predictions = []
references = []
methods_used = {'single_pass': 0, 'single_chunk': 0, 'map_reduce': 0}
chunk_counts = []

print("\nGenerating summaries...")
for i, item in enumerate(tqdm(test_data, desc="Processing")):
    article = item['text']
    reference = item['summary']
    
    # Generate summary
    result = summarizer.summarize(article)
    prediction = result['summary']
    method = result['method'] 
    num_chunks = result['num_chunks']
    
    predictions.append(prediction)
    references.append(reference)
    methods_used[method] += 1
    chunk_counts.append(num_chunks)
    
    # Show first few examples
    if i < 3:
        print(f"\n--- Sample {i+1} (Index {1600+i} in original test set) ---")
        print(f"Method: {method}, Chunks: {num_chunks}")
        print(f"Reference: {reference[:200]}...")
        print(f"Prediction: {prediction[:200]}...")

print("\n" + "="*80)
print("Computing metrics...")

# ROUGE scores
rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="ROUGE"):
    scores = scorer.score(ref, pred)
    rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
    rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
    rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)

# BERTScore
print("\nComputing BERTScore...")
P, R, F1 = bert_score_fn(predictions, references, lang='en', verbose=True)
bertscore_f1 = F1.mean().item()

# Semantic Similarity
print("\nComputing Semantic Similarity...")
pred_embeddings = semantic_model.encode(predictions, convert_to_tensor=True, show_progress_bar=True)
ref_embeddings = semantic_model.encode(references, convert_to_tensor=True, show_progress_bar=True)
semantic_sims = util.cos_sim(pred_embeddings, ref_embeddings).diagonal().cpu().numpy()

# BLEU scores
print("\nComputing BLEU...")
bleu_scores = []
smooth = SmoothingFunction()
for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="BLEU"):
    pred_tokens = pred.split()
    ref_tokens = [ref.split()]
    bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smooth.method1)
    bleu_scores.append(bleu)

# Aggregate results
results = {
    "dataset": "test_remaining_388.json",
    "num_samples": len(test_data),
    "original_indices": "1600-1987",
    "methods_distribution": methods_used,
    "avg_chunks": float(np.mean(chunk_counts)),
    "metrics": {
        "rouge1": float(np.mean(rouge_scores['rouge1'])),
        "rouge2": float(np.mean(rouge_scores['rouge2'])),
        "rougeL": float(np.mean(rouge_scores['rougeL'])),
        "bertscore_f1": float(bertscore_f1),
        "semantic_similarity": float(np.mean(semantic_sims)),
        "bleu": float(np.mean(bleu_scores))
    }
}

print("\n" + "="*80)
print("RESULTS FOR REMAINING 388 SAMPLES")
print("="*80)
print(f"Methods used:")
for method, count in methods_used.items():
    print(f"  {method}: {count} ({count/len(test_data)*100:.1f}%)")
print(f"\nAverage chunks: {results['avg_chunks']:.2f}")
print(f"\nMetrics:")
print(f"  ROUGE-1: {results['metrics']['rouge1']:.4f}")
print(f"  ROUGE-2: {results['metrics']['rouge2']:.4f}")
print(f"  ROUGE-L: {results['metrics']['rougeL']:.4f}")
print(f"  BERTScore F1: {results['metrics']['bertscore_f1']:.4f}")
print(f"  Semantic Similarity: {results['metrics']['semantic_similarity']:.4f}")
print(f"  BLEU: {results['metrics']['bleu']:.4f}")
print("="*80)

# Save results
with open('eval_remaining_388_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to eval_remaining_388_results.json")

# Save predictions for inspection
predictions_output = []
for i, (item, pred, ref) in enumerate(zip(test_data, predictions, references)):
    predictions_output.append({
        "original_index": 1600 + i,
        "text": item['text'],
        "reference": ref,
        "prediction": pred
    })

with open('remaining_388_predictions.json', 'w', encoding='utf-8') as f:
    json.dump(predictions_output, f, indent=2, ensure_ascii=False)

print(f"Predictions saved to remaining_388_predictions.json")
