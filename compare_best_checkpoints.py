"""
Compare best checkpoints from mT5-base and mT5-XLSum on FULL test dataset
- mT5-base best: checkpoint-16000 from mt5_teacher_mt5-base_20260208_124334
  (semantic_similarity=0.8976, best from previous evaluation)
- mT5-XLSum best: checkpoint-12000 from mt5_xlsum_20260212_060223
  (semantic_similarity=0.6778, best from xlsum evaluation)

Tests on entire test dataset with all metrics:
- ROUGE (standard summarization metric)
- Semantic Similarity / Cosine Similarity (MAIN TARGET)
- BERTScore (contextual similarity)
- BLEU (n-gram overlap)
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import warnings
warnings.filterwarnings('ignore')

# Configuration
CHECKPOINTS = [
    {
        "name": "mT5-base (checkpoint-16000)",
        "path": "mt5_teacher_mt5-base_20260208_124334/checkpoint-16000"
    },
    {
        "name": "mT5-XLSum (checkpoint-12000)",
        "path": "mt5_xlsum_20260212_060223/checkpoint-12000"
    }
]

TEST_FILE = "data/test.csv"
OUTPUT_FILE = "best_checkpoints_comparison_full_test.json"

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Custom tokenizer for Bangla ROUGE
class SpaceTokenizer:
    def tokenize(self, text):
        return text.split()

print("="*80)
print("COMPARING BEST CHECKPOINTS ON FULL TEST DATASET")
print("="*80)

# Load FULL test data
print(f"\n📊 Loading FULL test dataset from: {TEST_FILE}")
test_df = pd.read_csv(TEST_FILE)
print(f"   Total test samples: {len(test_df)}")
print(f"   Using ALL samples for accurate comparison")

test_texts = test_df['text'].tolist()
test_summaries = test_df['summary'].tolist()

# Initialize metrics
print("\n🔧 Initializing metrics...")
print("   - ROUGE scorer")
rouge_scorer_obj = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'], 
    use_stemmer=False, 
    tokenizer=SpaceTokenizer()
)

print("   - Semantic similarity model (sentence-transformers)")
try:
    semantic_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    semantic_model.to(device)
    semantic_available = True
except Exception as e:
    print(f"      ⚠️  Could not load semantic model: {e}")
    semantic_available = False

print("   - BERTScore")
try:
    from bert_score import score as bert_score
    bertscore_available = True
except:
    print("      ⚠️  BERTScore not available (install: pip install bert-score)")
    bertscore_available = False

print("   - BLEU score")
try:
    from sacrebleu.metrics import BLEU
    bleu_metric = BLEU()
    bleu_available = True
except:
    print("      ⚠️  BLEU not available (install: pip install sacrebleu)")
    bleu_available = False

# Evaluate each checkpoint
results = []

for checkpoint_info in CHECKPOINTS:
    checkpoint_name = checkpoint_info["name"]
    checkpoint_path = checkpoint_info["path"]
    
    print(f"\n{'='*80}")
    print(f"🔍 Evaluating: {checkpoint_name}")
    print(f"   Path: {checkpoint_path}")
    print('='*80)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        continue
    
    try:
        # Load model and tokenizer
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
        model.to(device)
        model.eval()
        
        # Generate summaries
        print(f"Generating summaries for {len(test_texts)} samples...")
        generated_summaries = []
        
        batch_size = 8
        for i in tqdm(range(0, len(test_texts), batch_size), desc="Generating"):
            batch_texts = test_texts[i:i+batch_size]
            
            inputs = tokenizer(
                batch_texts,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_summaries.extend(decoded)
        
        # Clean up model to save memory
        del model
        torch.cuda.empty_cache()
        
        # Compute metrics
        print("\nComputing metrics...")
        
        # 1. ROUGE scores
        print("   - ROUGE...")
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for ref, pred in zip(test_summaries, generated_summaries):
            scores = rouge_scorer_obj.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        avg_rouge1 = np.mean(rouge1_scores)
        avg_rouge2 = np.mean(rouge2_scores)
        avg_rougeL = np.mean(rougeL_scores)
        
        # 2. Semantic Similarity (MAIN TARGET)
        if semantic_available:
            print("   - Semantic Similarity (cosine)...")
            # Process in batches to avoid memory issues
            batch_size_embed = 32
            cosine_scores = []
            
            for i in tqdm(range(0, len(test_summaries), batch_size_embed), desc="Computing similarity"):
                batch_refs = test_summaries[i:i+batch_size_embed]
                batch_preds = generated_summaries[i:i+batch_size_embed]
                
                ref_embeddings = semantic_model.encode(batch_refs, convert_to_tensor=True, show_progress_bar=False)
                pred_embeddings = semantic_model.encode(batch_preds, convert_to_tensor=True, show_progress_bar=False)
                
                batch_scores = util.cos_sim(ref_embeddings, pred_embeddings).diagonal().cpu().numpy()
                cosine_scores.extend(batch_scores)
            
            avg_semantic_sim = np.mean(cosine_scores)
        else:
            avg_semantic_sim = None
        
        # 3. BERTScore
        if bertscore_available:
            print("   - BERTScore...")
            try:
                # Process in batches for large dataset
                batch_size_bert = 64
                P_scores, R_scores, F1_scores = [], [], []
                
                for i in tqdm(range(0, len(test_summaries), batch_size_bert), desc="Computing BERTScore"):
                    batch_preds = generated_summaries[i:i+batch_size_bert]
                    batch_refs = test_summaries[i:i+batch_size_bert]
                    
                    P, R, F1 = bert_score(batch_preds, batch_refs, lang='other', verbose=False)
                    P_scores.extend(P.tolist())
                    R_scores.extend(R.tolist())
                    F1_scores.extend(F1.tolist())
                
                avg_bertscore_f1 = np.mean(F1_scores)
            except Exception as e:
                print(f"      Error computing BERTScore: {e}")
                avg_bertscore_f1 = None
        else:
            avg_bertscore_f1 = None
        
        # 4. BLEU score
        if bleu_available:
            print("   - BLEU...")
            try:
                refs_formatted = [[ref] for ref in test_summaries]
                bleu_result = bleu_metric.corpus_score(generated_summaries, refs_formatted)
                avg_bleu = bleu_result.score
            except Exception as e:
                print(f"      Error computing BLEU: {e}")
                avg_bleu = None
        else:
            avg_bleu = None
        
        # Store results
        result = {
            'checkpoint': checkpoint_name,
            'path': checkpoint_path,
            'test_samples': len(test_texts),
            'rouge1': float(avg_rouge1),
            'rouge2': float(avg_rouge2),
            'rougeL': float(avg_rougeL),
            'semantic_similarity': float(avg_semantic_sim) if avg_semantic_sim is not None else None,
            'bertscore_f1': float(avg_bertscore_f1) if avg_bertscore_f1 is not None else None,
            'bleu': float(avg_bleu) if avg_bleu is not None else None,
        }
        results.append(result)
        
        # Print results
        print(f"\n📊 Results for {checkpoint_name}:")
        print(f"   ROUGE-1: {avg_rouge1:.4f}")
        print(f"   ROUGE-2: {avg_rouge2:.4f}")
        print(f"   ROUGE-L: {avg_rougeL:.4f}")
        if avg_semantic_sim is not None:
            print(f"   🎯 Semantic Similarity: {avg_semantic_sim:.4f}")
        if avg_bertscore_f1 is not None:
            print(f"   BERTScore-F1: {avg_bertscore_f1:.4f}")
        if avg_bleu is not None:
            print(f"   BLEU: {avg_bleu:.2f}")
        
    except Exception as e:
        print(f"❌ Error evaluating {checkpoint_name}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Save results
print(f"\n{'='*80}")
print("SAVING RESULTS")
print('='*80)

with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✅ Results saved to: {OUTPUT_FILE}")

# Comparison table
print(f"\n{'='*80}")
print("COMPARISON TABLE (FULL TEST DATASET)")
print('='*80)

if len(results) >= 2:
    print(f"\nTest samples: {results[0]['test_samples']}")
    print(f"\n{'Model':<30} {'ROUGE-L':<10} {'🎯 Semantic':<12} {'BERTScore':<12} {'BLEU':<10}")
    print("-"*80)
    
    for result in results:
        semantic_str = f"{result['semantic_similarity']:.4f}" if result['semantic_similarity'] is not None else "N/A"
        bert_str = f"{result['bertscore_f1']:.4f}" if result['bertscore_f1'] is not None else "N/A"
        bleu_str = f"{result['bleu']:.2f}" if result['bleu'] is not None else "N/A"
        print(f"{result['checkpoint']:<30} {result['rougeL']:<10.4f} {semantic_str:<12} {bert_str:<12} {bleu_str:<10}")
    
    # Calculate improvements
    print(f"\n{'='*80}")
    print("IMPROVEMENT ANALYSIS")
    print('='*80)
    
    base_result = results[0]  # mT5-base
    xlsum_result = results[1]  # mT5-XLSum
    
    rouge_improvement = ((xlsum_result['rougeL'] - base_result['rougeL']) / base_result['rougeL']) * 100
    print(f"\n📈 mT5-XLSum vs mT5-base:")
    print(f"   ROUGE-L: {base_result['rougeL']:.4f} → {xlsum_result['rougeL']:.4f} ({rouge_improvement:+.1f}%)")
    
    if xlsum_result['semantic_similarity'] is not None and base_result['semantic_similarity'] is not None:
        semantic_improvement = ((xlsum_result['semantic_similarity'] - base_result['semantic_similarity']) / base_result['semantic_similarity']) * 100
        print(f"   🎯 Semantic Similarity: {base_result['semantic_similarity']:.4f} → {xlsum_result['semantic_similarity']:.4f} ({semantic_improvement:+.1f}%)")
    
    if xlsum_result['bertscore_f1'] is not None and base_result['bertscore_f1'] is not None:
        bert_improvement = ((xlsum_result['bertscore_f1'] - base_result['bertscore_f1']) / base_result['bertscore_f1']) * 100
        print(f"   BERTScore-F1: {base_result['bertscore_f1']:.4f} → {xlsum_result['bertscore_f1']:.4f} ({bert_improvement:+.1f}%)")
    
    if xlsum_result['bleu'] is not None and base_result['bleu'] is not None:
        bleu_improvement = ((xlsum_result['bleu'] - base_result['bleu']) / base_result['bleu']) * 100
        print(f"   BLEU: {base_result['bleu']:.2f} → {xlsum_result['bleu']:.2f} ({bleu_improvement:+.1f}%)")
    
    # Winner determination
    print(f"\n{'='*80}")
    print("WINNER")
    print('='*80)
    
    if xlsum_result['semantic_similarity'] is not None and base_result['semantic_similarity'] is not None:
        if xlsum_result['semantic_similarity'] > base_result['semantic_similarity']:
            winner = xlsum_result['checkpoint']
            winner_metric = xlsum_result['semantic_similarity']
        else:
            winner = base_result['checkpoint']
            winner_metric = base_result['semantic_similarity']
        
        print(f"\n🏆 BEST MODEL (by Semantic Similarity - MAIN TARGET):")
        print(f"   {winner}")
        print(f"   Semantic Similarity: {winner_metric:.4f}")
    else:
        if xlsum_result['rougeL'] > base_result['rougeL']:
            winner = xlsum_result['checkpoint']
            winner_metric = xlsum_result['rougeL']
        else:
            winner = base_result['checkpoint']
            winner_metric = base_result['rougeL']
        
        print(f"\n🏆 BEST MODEL (by ROUGE-L):")
        print(f"   {winner}")
        print(f"   ROUGE-L: {winner_metric:.4f}")

print(f"\n{'='*80}")
print("✅ COMPARISON COMPLETE!")
print('='*80)
