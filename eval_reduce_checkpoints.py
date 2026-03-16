"""
Evaluate all reduce task checkpoints on reduce_test.json
Finds the best checkpoint by ROUGE-L, BERTScore, and Semantic Similarity.
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────
CHECKPOINT_DIR = "banglaT5_reduce_task_20260217_111025"
TEST_FILE = "reduce_data/reduce_test.json"
NUM_TEST_SAMPLES = 500          # use 500 for speed; set higher for final
OUTPUT_FILE = "reduce_checkpoint_evaluation.json"
INPUT_PREFIX = "summarize multiple summaries: "
BATCH_SIZE = 8

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Bangla fix ─────────────────────────────────────────────────
def fix_bangla_for_tokenizer(text):
    text = text.replace('\u09DF', '\u09AF\u09BC')
    text = text.replace('\u09DC', '\u09A1\u09BC')
    text = text.replace('\u09DD', '\u09A2\u09BC')
    text = text.replace('\u200c', '').replace('\u200d', '')
    return text

# ── Word-level ROUGE tokenizer ─────────────────────────────────
class SpaceTokenizer:
    def tokenize(self, text):
        return text.split()

# ── Discover checkpoints ───────────────────────────────────────
# Only evaluate the top candidates (from partial run before power outage)
ONLY_THESE = ["checkpoint-2000", "checkpoint-5500", "checkpoint-6000", "final_model"]

print(f"\nScanning: {CHECKPOINT_DIR}")
checkpoint_dirs = []
for item in sorted(os.listdir(CHECKPOINT_DIR)):
    path = os.path.join(CHECKPOINT_DIR, item)
    if os.path.isdir(path) and item in ONLY_THESE:
        checkpoint_dirs.append(path)

print(f"Found {len(checkpoint_dirs)} checkpoints:")
for cp in checkpoint_dirs:
    size_mb = sum(f.stat().st_size for f in Path(cp).rglob('*') if f.is_file()) / (1024*1024)
    print(f"  {os.path.basename(cp):<20} {size_mb:.0f} MB")

# ── Load test data ─────────────────────────────────────────────
print(f"\nLoading test data: {TEST_FILE}")
with open(TEST_FILE, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# Only use 'clean' augmentation samples for fair eval
clean_data = [d for d in test_data if d.get('augmentation', 'clean') == 'clean']
print(f"  Total: {len(test_data)}, Clean only: {len(clean_data)}")

import random
random.seed(42)
sample = random.sample(clean_data, min(NUM_TEST_SAMPLES, len(clean_data)))

texts = [INPUT_PREFIX + fix_bangla_for_tokenizer(d['text']) for d in sample]
refs  = [fix_bangla_for_tokenizer(d['summary']) for d in sample]
print(f"  Evaluating on {len(sample)} clean samples")

# ── Metric initialisers ───────────────────────────────────────
rouge_obj = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=False,
    tokenizer=SpaceTokenizer()
)

print("Loading semantic model...")
sem_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
sem_model.to(device)

from bert_score import score as bert_score_fn

# ── Evaluate each checkpoint ──────────────────────────────────
results = []

for cp_path in checkpoint_dirs:
    cp_name = os.path.basename(cp_path)
    print(f"\n{'='*70}")
    print(f"Evaluating: {cp_name}")
    print('='*70)

    tokenizer = AutoTokenizer.from_pretrained(cp_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(cp_path)
    model.to(device).eval()

    # Generate summaries
    preds = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Generating"):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, max_length=1024, truncation=True,
                           padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=256, num_beams=5,
                                 early_stopping=True, no_repeat_ngram_size=3,
                                 length_penalty=1.0, repetition_penalty=1.2)
        preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))

    del model
    torch.cuda.empty_cache()

    # ROUGE
    r1, r2, rL = [], [], []
    for ref, pred in zip(refs, preds):
        s = rouge_obj.score(ref, pred)
        r1.append(s['rouge1'].fmeasure)
        r2.append(s['rouge2'].fmeasure)
        rL.append(s['rougeL'].fmeasure)

    # Semantic similarity
    ref_emb  = sem_model.encode(refs,  convert_to_tensor=True, show_progress_bar=False)
    pred_emb = sem_model.encode(preds, convert_to_tensor=True, show_progress_bar=False)
    sem_scores = util.cos_sim(ref_emb, pred_emb).diagonal().cpu().numpy()

    # BERTScore
    _, _, bert_f1 = bert_score_fn(preds, refs, lang='other', verbose=False)

    row = {
        'checkpoint': cp_name,
        'path': cp_path,
        'rouge1': float(np.mean(r1)),
        'rouge2': float(np.mean(r2)),
        'rougeL': float(np.mean(rL)),
        'semantic': float(np.mean(sem_scores)),
        'bertscore': float(bert_f1.mean().item()),
    }
    results.append(row)

    print(f"  ROUGE-1: {row['rouge1']:.4f}  ROUGE-2: {row['rouge2']:.4f}  "
          f"ROUGE-L: {row['rougeL']:.4f}")
    print(f"  Semantic: {row['semantic']:.4f}  BERTScore: {row['bertscore']:.4f}")

# ── Save results ───────────────────────────────────────────────
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUTPUT_FILE}")

# ── Summary table ──────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"{'Checkpoint':<20} {'ROUGE-L':>8} {'Semantic':>9} {'BERTScore':>10}")
print('-'*70)
for r in results:
    print(f"{r['checkpoint']:<20} {r['rougeL']:>8.4f} {r['semantic']:>9.4f} {r['bertscore']:>10.4f}")

# Best by each metric
best_rl = max(results, key=lambda x: x['rougeL'])
best_sem = max(results, key=lambda x: x['semantic'])
best_bs = max(results, key=lambda x: x['bertscore'])

print(f"\nBest ROUGE-L:   {best_rl['checkpoint']}  ({best_rl['rougeL']:.4f})")
print(f"Best Semantic:  {best_sem['checkpoint']}  ({best_sem['semantic']:.4f})")
print(f"Best BERTScore: {best_bs['checkpoint']}  ({best_bs['bertscore']:.4f})")

# Overall best (average rank)
for r in results:
    rl_rank = sorted(results, key=lambda x: -x['rougeL']).index(r)
    sem_rank = sorted(results, key=lambda x: -x['semantic']).index(r)
    bs_rank = sorted(results, key=lambda x: -x['bertscore']).index(r)
    r['avg_rank'] = (rl_rank + sem_rank + bs_rank) / 3

overall_best = min(results, key=lambda x: x['avg_rank'])
print(f"\nOVERALL BEST (avg rank): {overall_best['checkpoint']}")
print(f"  ROUGE-L: {overall_best['rougeL']:.4f}  Semantic: {overall_best['semantic']:.4f}  "
      f"BERTScore: {overall_best['bertscore']:.4f}")
