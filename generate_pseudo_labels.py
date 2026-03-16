"""
Generate Pseudo-Labels from mT5 Teacher Models
===============================================
Generates summaries from mT5-base and mT5-XLSum checkpoints for use in
multi-teacher knowledge distillation. These serve as sequence-level
knowledge transfer from cross-architecture teachers.

Run this BEFORE running the ablation study.
Usage: python generate_pseudo_labels.py [--quick]
"""

import os
import json
import gc
import torch
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ============================================================================
# Configuration
# ============================================================================

TEACHERS = {
    "mt5_base": {
        "path": "./mt5_teacher_mt5-base_20260208_124334/checkpoint-16000",
        "tokenizer": "google/mt5-base",
        "name": "mT5-base",
    },
    "mt5_xlsum": {
        "path": "./mt5_xlsum_20260212_060223/checkpoint-12000",
        "tokenizer": "csebuetnlp/mT5_multilingual_XLSum",
        "name": "mT5-XLSum",
    },
}

TRAIN_FILE = "data/train.csv"
OUTPUT_DIR = "data/pseudo_labels"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128   # mT5 teachers average ~32 tokens output
BATCH_SIZE = 16           # Reduced from 32 to avoid OOM on long sequences
NUM_BEAMS = 4


def generate_summaries(model_path, tokenizer_name, model_name, texts, batch_size=16, checkpoint_file=None):
    """Generate summaries for a list of texts using a teacher model.
    Saves progress every 50 batches to checkpoint_file for crash recovery."""
    print(f"\n  Loading {model_name} from {model_path}...")
    print(f"  Tokenizer from {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    all_summaries = []
    
    # Resume support: load partial results if checkpoint exists
    partial_file = (checkpoint_file or "partial") + ".partial.json"
    resume_from = 0
    if os.path.exists(partial_file):
        with open(partial_file, "r", encoding="utf-8") as f:
            all_summaries = json.load(f)
        resume_from = len(all_summaries)
        print(f"  Resuming from sample {resume_from}/{len(texts)} (loaded partial checkpoint)")

    SAVE_EVERY = 50  # Save checkpoint every 50 batches

    for batch_idx, i in enumerate(tqdm(range(0, len(texts), batch_size), desc=f"  {model_name}")):
        if i < resume_from:
            continue
            
        batch_texts = [str(t) for t in texts[i : i + batch_size]]

        try:
            inputs = tokenizer(
                batch_texts,
                max_length=MAX_INPUT_LENGTH,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=MAX_TARGET_LENGTH,
                    num_beams=NUM_BEAMS,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_summaries.extend(decoded)
            
        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM at batch {i//batch_size}! Falling back to batch_size=1...")
            torch.cuda.empty_cache()
            # Process this batch one-by-one
            for t in batch_texts:
                try:
                    inp = tokenizer(
                        [t], max_length=MAX_INPUT_LENGTH,
                        truncation=True, padding=True, return_tensors="pt",
                    ).to(device)
                    with torch.no_grad():
                        out = model.generate(
                            **inp, max_length=MAX_TARGET_LENGTH,
                            num_beams=NUM_BEAMS, early_stopping=True,
                            no_repeat_ngram_size=3,
                        )
                    all_summaries.append(tokenizer.decode(out[0], skip_special_tokens=True))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"\n  OOM even on single sample, using empty string")
                    all_summaries.append("")

        # Periodic checkpoint save
        if batch_idx > 0 and batch_idx % SAVE_EVERY == 0:
            with open(partial_file, "w", encoding="utf-8") as f:
                json.dump(all_summaries, f, ensure_ascii=False)
            tqdm.write(f"  [Checkpoint] Saved {len(all_summaries)}/{len(texts)} samples")

    # Cleanup partial checkpoint after successful completion
    if os.path.exists(partial_file):
        os.remove(partial_file)

    # Cleanup GPU memory
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return all_summaries


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-labels from mT5 teachers")
    parser.add_argument("--quick", action="store_true", help="Quick test with 200 samples")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("PSEUDO-LABEL GENERATION FOR MULTI-TEACHER DISTILLATION")
    print("=" * 80)

    print(f"\nLoading training data from {TRAIN_FILE}...")
    df = pd.read_csv(TRAIN_FILE)
    texts = df["text"].tolist()

    if args.quick:
        texts = texts[:200]
        print(f"[QUICK MODE] Using only {len(texts)} samples")
    else:
        print(f"Total samples: {len(texts)}")

    for teacher_key, teacher_info in TEACHERS.items():
        output_file = os.path.join(OUTPUT_DIR, f"train_{teacher_key}.json")

        # Check if already generated
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if len(existing) == len(texts):
                print(f"\n  Already exists: {output_file} ({len(existing)} samples)")
                continue
            else:
                print(f"\n  Existing file has {len(existing)} samples, need {len(texts)}. Regenerating...")

        print(f"\n  Generating with {teacher_info['name']}...")
        summaries = generate_summaries(
            teacher_info["path"],
            teacher_info["tokenizer"],
            teacher_info["name"],
            texts,
            batch_size=BATCH_SIZE,
            checkpoint_file=output_file,
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False)

        print(f"  Saved {len(summaries)} pseudo-labels to {output_file}")

    print(f"\n{'='*80}")
    print("PSEUDO-LABEL GENERATION COMPLETE!")
    print(f"{'='*80}")
    for teacher_key in TEACHERS:
        fpath = f"{OUTPUT_DIR}/train_{teacher_key}.json"
        exists = os.path.exists(fpath)
        print(f"  {'OK' if exists else 'MISSING'} {fpath}")


if __name__ == "__main__":
    main()
