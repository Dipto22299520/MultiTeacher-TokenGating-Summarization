"""
Comprehensive checkpoint evaluation script
Tests all checkpoints on test set with multiple metrics:
- ROUGE (standard summarization metric)
- Semantic Similarity / Cosine Similarity (MAIN TARGET)
- BERTScore (contextual similarity)
- BLEU (n-gram overlap)

Finds best checkpoint and helps free up disk space.
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
CHECKPOINT_DIR = "mt5_teacher_mt5-base_20260208_124334"
TEST_FILE = "data/test.csv"
NUM_TEST_SAMPLES = 1000  # Evaluate on 1000 samples for speed (can increase)
OUTPUT_FILE = "checkpoint_evaluation_results.json"

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Custom tokenizer for Bangla ROUGE
class SpaceTokenizer:
    def tokenize(self, text):
        return text.split()

print("="*80)
print("CHECKPOINT EVALUATION - FINDING BEST MODEL")
print("="*80)

# Find all checkpoints
print(f"\n📁 Scanning directory: {CHECKPOINT_DIR}")
checkpoint_dirs = []
if os.path.exists(CHECKPOINT_DIR):
    for item in os.listdir(CHECKPOINT_DIR):
        if item.startswith('checkpoint-'):
            checkpoint_path = os.path.join(CHECKPOINT_DIR, item)
            if os.path.isdir(checkpoint_path):
                checkpoint_dirs.append(checkpoint_path)

# Also check final_model
final_model_path = os.path.join(CHECKPOINT_DIR, "final_model")
if os.path.exists(final_model_path):
    checkpoint_dirs.append(final_model_path)

checkpoint_dirs = sorted(checkpoint_dirs)
print(f"\n✅ Found {len(checkpoint_dirs)} checkpoints:")
for cp in checkpoint_dirs:
    size_mb = sum(f.stat().st_size for f in Path(cp).rglob('*') if f.is_file()) / 1024 / 1024
    print(f"   - {os.path.basename(cp):<20} ({size_mb:.1f} MB)")

if not checkpoint_dirs:
    print("❌ No checkpoints found!")
    exit(1)

# Load test data
print(f"\n📊 Loading test data from: {TEST_FILE}")
test_df = pd.read_csv(TEST_FILE)
print(f"   Total test samples: {len(test_df)}")
print(f"   Evaluating on: {NUM_TEST_SAMPLES} samples")

# Sample test data
test_df = test_df.sample(n=min(NUM_TEST_SAMPLES, len(test_df)), random_state=42)
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
    # Use multilingual model for Bangla support
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

for checkpoint_path in checkpoint_dirs:
    checkpoint_name = os.path.basename(checkpoint_path)
    print(f"\n{'='*80}")
    print(f"🔍 Evaluating: {checkpoint_name}")
    print('='*80)
    
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
            ref_embeddings = semantic_model.encode(test_summaries, convert_to_tensor=True, show_progress_bar=False)
            pred_embeddings = semantic_model.encode(generated_summaries, convert_to_tensor=True, show_progress_bar=False)
            
            cosine_scores = util.cos_sim(ref_embeddings, pred_embeddings).diagonal().cpu().numpy()
            avg_semantic_sim = np.mean(cosine_scores)
        else:
            avg_semantic_sim = None
        
        # 3. BERTScore
        if bertscore_available:
            print("   - BERTScore...")
            try:
                P, R, F1 = bert_score(generated_summaries, test_summaries, lang='other', verbose=False)
                avg_bertscore_f1 = F1.mean().item()
            except Exception as e:
                print(f"      Error computing BERTScore: {e}")
                avg_bertscore_f1 = None
        else:
            avg_bertscore_f1 = None
        
        # 4. BLEU score
        if bleu_available:
            print("   - BLEU...")
            try:
                # BLEU expects list of references per prediction
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
            print(f"   BLEU: {avg_bleu:.4f}")
        
    except Exception as e:
        print(f"❌ Error evaluating {checkpoint_name}: {e}")
        continue

# Save results
print(f"\n{'='*80}")
print("SAVING RESULTS")
print('='*80)

with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✅ Results saved to: {OUTPUT_FILE}")

# Summary table
print(f"\n{'='*80}")
print("SUMMARY TABLE")
print('='*80)

results_df = pd.DataFrame(results)
print(f"\n{'Checkpoint':<25} {'ROUGE-L':<10} {'🎯 Semantic':<12} {'BERTScore':<12} {'BLEU':<10}")
print("-"*80)
for _, row in results_df.iterrows():
    semantic_str = f"{row['semantic_similarity']:.4f}" if row['semantic_similarity'] is not None else "N/A"
    bert_str = f"{row['bertscore_f1']:.4f}" if row['bertscore_f1'] is not None else "N/A"
    bleu_str = f"{row['bleu']:.4f}" if row['bleu'] is not None else "N/A"
    print(f"{row['checkpoint']:<25} {row['rougeL']:<10.4f} {semantic_str:<12} {bert_str:<12} {bleu_str:<10}")

# Find best checkpoint
print(f"\n{'='*80}")
print("BEST CHECKPOINTS BY METRIC")
print('='*80)

if 'semantic_similarity' in results_df.columns and results_df['semantic_similarity'].notna().any():
    best_semantic = results_df.loc[results_df['semantic_similarity'].idxmax()]
    print(f"\n🎯 BEST by Semantic Similarity (MAIN TARGET):")
    print(f"   Checkpoint: {best_semantic['checkpoint']}")
    print(f"   Semantic Similarity: {best_semantic['semantic_similarity']:.4f}")
    print(f"   ROUGE-L: {best_semantic['rougeL']:.4f}")

best_rougeL = results_df.loc[results_df['rougeL'].idxmax()]
print(f"\n📊 BEST by ROUGE-L:")
print(f"   Checkpoint: {best_rougeL['checkpoint']}")
print(f"   ROUGE-L: {best_rougeL['rougeL']:.4f}")
if best_rougeL['semantic_similarity'] is not None:
    print(f"   Semantic Similarity: {best_rougeL['semantic_similarity']:.4f}")

if 'bertscore_f1' in results_df.columns and results_df['bertscore_f1'].notna().any():
    best_bert = results_df.loc[results_df['bertscore_f1'].idxmax()]
    print(f"\n🤖 BEST by BERTScore:")
    print(f"   Checkpoint: {best_bert['checkpoint']}")
    print(f"   BERTScore-F1: {best_bert['bertscore_f1']:.4f}")

# Recommendation
print(f"\n{'='*80}")
print("RECOMMENDATION & DISK SPACE CLEANUP")
print('='*80)

if 'semantic_similarity' in results_df.columns and results_df['semantic_similarity'].notna().any():
    best_checkpoint = best_semantic['checkpoint']
    best_path = best_semantic['path']
else:
    best_checkpoint = best_rougeL['checkpoint']
    best_path = best_rougeL['path']

print(f"\n✅ RECOMMENDED CHECKPOINT: {best_checkpoint}")
print(f"   Path: {best_path}")

# Calculate disk space
total_space = 0
other_checkpoints = []
for result in results:
    cp_path = result['path']
    size_mb = sum(f.stat().st_size for f in Path(cp_path).rglob('*') if f.is_file()) / 1024 / 1024
    total_space += size_mb
    
    if result['checkpoint'] != best_checkpoint:
        other_checkpoints.append((result['checkpoint'], cp_path, size_mb))

space_to_free = sum(size for _, _, size in other_checkpoints)

print(f"\n💾 Disk Space Analysis:")
print(f"   Total space used: {total_space:.1f} MB")
print(f"   Best checkpoint: {sum(f.stat().st_size for f in Path(best_path).rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")
print(f"   Other checkpoints: {space_to_free:.1f} MB ({len(other_checkpoints)} checkpoints)")

print(f"\n🗑️  Checkpoints that can be deleted:")
for name, path, size in other_checkpoints:
    print(f"   - {name:<25} ({size:.1f} MB)")

print(f"\n⚠️  After cleanup: Free up ~{space_to_free:.1f} MB ({space_to_free/1024:.2f} GB)")

print(f"\n{'='*80}")
print("MANUAL CLEANUP INSTRUCTIONS")
print('='*80)
print(f"""
To free up {space_to_free/1024:.2f} GB, delete these checkpoints manually:

Windows (PowerShell):
""")
for name, path, _ in other_checkpoints:
    print(f'   Remove-Item -Recurse -Force "{path}"')

print(f"""
Or keep them all for comparison and delete later after final testing.
""")
