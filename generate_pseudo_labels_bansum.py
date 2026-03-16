"""
Generate Pseudo-Labels from mT5 BanSum Teacher Models
=====================================================
Generates summaries from mT5-base-bansum and mT5-xlsum-bansum checkpoints
for use in multi-teacher knowledge distillation on BanSum dataset.

Run this BEFORE running the ablation study.
Usage: python generate_pseudo_labels_bansum.py [--quick]
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
        "path": "./mt5base_bansum_20260219_113113/checkpoint-16000",
        "tokenizer": "google/mt5-base",
        "name": "mT5-base (BanSum)",
    },
    "mt5_xlsum": {
        "path": "./mt5xlsum_bansum_20260219_062938/checkpoint-14000",
        "tokenizer": "csebuetnlp/mT5_multilingual_XLSum",
        "name": "mT5-XLSum (BanSum)",
    },
}

BANSUM_FILE = "bansum_lte_1000_tokens.json"
OUTPUT_DIR = "data/pseudo_labels_bansum"
SEED = 42

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 32
NUM_BEAMS = 4


def load_bansum_train_texts():
    """Load BanSum dataset and return ONLY training split texts (80/10/10)."""
    print(f"\nLoading BanSum data from {BANSUM_FILE}...")
    with open(BANSUM_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame([{"text": item["main"], "summary": item["sum2"]} for item in data])
    print(f"  Total samples: {len(df)}")

    # Same split as train_teacher_bansum.py
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    print(f"  Training set: {len(train_df)} samples")
    return train_df["text"].tolist()


def generate_summaries(model_path, tokenizer_name, model_name, texts, batch_size=16, checkpoint_file=None):
    """Generate summaries for a list of texts using a teacher model."""
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
        print(f"  Resuming from sample {resume_from}/{len(texts)}")

    SAVE_EVERY = 50

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

    # Cleanup partial checkpoint
    if os.path.exists(partial_file):
        os.remove(partial_file)

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return all_summaries


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-labels from BanSum mT5 teachers")
    parser.add_argument("--quick", action="store_true", help="Quick test with 200 samples")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("PSEUDO-LABEL GENERATION FOR BANSUM MULTI-TEACHER DISTILLATION")
    print("=" * 80)

    texts = load_bansum_train_texts()

    if args.quick:
        texts = texts[:200]
        print(f"[QUICK MODE] Using only {len(texts)} samples")

    for teacher_key, teacher_info in TEACHERS.items():
        output_file = os.path.join(OUTPUT_DIR, f"train_{teacher_key}.json")

        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if len(existing) == len(texts):
                print(f"\n  Already exists: {output_file} ({len(existing)} samples)")
                continue
            else:
                print(f"\n  Existing has {len(existing)}, need {len(texts)}. Regenerating...")

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
    print("PSEUDO-LABEL GENERATION (BANSUM) COMPLETE!")
    print(f"{'='*80}")
    for teacher_key in TEACHERS:
        fpath = f"{OUTPUT_DIR}/train_{teacher_key}.json"
        exists = os.path.exists(fpath)
        print(f"  {'OK' if exists else 'MISSING'} {fpath}")


if __name__ == "__main__":
    main()
