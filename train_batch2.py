"""
Training pipeline for batch 2 languages:
  Ukrainian, Tamil, Telugu, Gujarati, Vietnamese

Steps per language:
  1. Fine-tune teacher  (csebuetnlp/mT5_multilingual_XLSum)
  2. Generate pseudo-labels
  3. Train student      (google/mt5-small)

Prerequisites:
  python preprocess_batch2.py
"""

import glob
import os
import re
import subprocess
import time
from datetime import datetime

LANGUAGES = ["ukrainian", "tamil", "telugu", "gujarati", "vietnamese"]

# If you already have a trained teacher, put its path here to skip retraining.
EXISTING_TEACHERS = {}


def update_config_file(filepath, updates):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    for pattern, replacement in updates:
        content = re.sub(pattern, lambda _m, r=replacement: r, content, flags=re.MULTILINE)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def find_latest_teacher(language):
    pattern = f"./teachers/{language}_teacher_*/final_model"
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No teacher model found for '{language}' matching: {pattern}")
    return max(matches, key=os.path.getmtime)


def run_command(cmd, description):
    print(f"\n{'=' * 70}")
    print(description)
    print(f"{'=' * 70}")
    print(f"Command : {cmd}")
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    elapsed = time.time() - start
    print(f"Finished in {elapsed / 60:.1f} min | return code: {result.returncode}")
    if result.returncode != 0:
        print(f"ERROR: command failed (rc={result.returncode})")
        return False
    return True


def train_language(language):
    print(f"\n\n{'#' * 70}")
    print(f"# STARTING: {language.upper()}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    for f in [
        f"./preprocessed_data/{language}/train.csv",
        f"./preprocessed_data/{language}/val.csv",
        f"./preprocessed_data/{language}/test.csv",
    ]:
        if not os.path.exists(f):
            print(f"ERROR: Missing {f} — run preprocess_batch2.py first.")
            return False

    # Step 1: train or reuse teacher
    teacher_model_path = EXISTING_TEACHERS.get(language)
    if teacher_model_path:
        if not os.path.exists(teacher_model_path):
            print(f"ERROR: Existing teacher not found: {teacher_model_path}")
            return False
        print(f"Using existing teacher: {teacher_model_path}")
    else:
        update_config_file("train_teacher.py", [
            (r'TRAIN_FILE = ".*"',  f'TRAIN_FILE = "./preprocessed_data/{language}/train.csv"'),
            (r'VAL_FILE = ".*"',    f'VAL_FILE = "./preprocessed_data/{language}/val.csv"'),
            (r'TEST_FILE = ".*"',   f'TEST_FILE = "./preprocessed_data/{language}/test.csv"'),
            (r'OUTPUT_DIR = ".*"',  f'OUTPUT_DIR = "./teachers/{language}_teacher"'),
        ])
        if not run_command("python train_teacher.py", f"[{language}] 1/3 Train teacher"):
            return False
        try:
            teacher_model_path = find_latest_teacher(language)
            print(f"Teacher: {teacher_model_path}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return False

    teacher_model_path = teacher_model_path.replace("\\", "/")

    # Step 2: generate pseudo-labels
    update_config_file("generate_teacher_labels.py", [
        (r'TEACHER_MODEL_PATH = ".*"',
         f'TEACHER_MODEL_PATH = "{teacher_model_path}"'),
        (r'TRAIN_FILE = ".*preprocessed_data/[^/]+/train\.csv"',
         f'TRAIN_FILE = "./preprocessed_data/{language}/train.csv"'),
        (r'VAL_FILE = ".*preprocessed_data/[^/]+/val\.csv"',
         f'VAL_FILE = "./preprocessed_data/{language}/val.csv"'),
        (r'TEST_FILE = ".*preprocessed_data/[^/]+/test\.csv"',
         f'TEST_FILE = "./preprocessed_data/{language}/test.csv"'),
        (r'OUTPUT_DIR = ".*preprocessed_data/.*"',
         f'OUTPUT_DIR = "./preprocessed_data/{language}_finetuned_teacher_labels"'),
        (r'USE_PREFIX = (True|False).*',
         'USE_PREFIX = True  # mT5 teachers were trained with prefix'),
    ])
    if not run_command("python generate_teacher_labels.py", f"[{language}] 2/3 Generate pseudo-labels"):
        return False

    # Step 3: train student
    update_config_file("train_student_fast.py", [
        (r'OUTPUT_DIR = ".*students.*"',
         f'OUTPUT_DIR = "./students/{language}_student_fast"'),
        (r'TRAIN_FILE = ".*preprocessed_data/.*labels/train\.csv"',
         f'TRAIN_FILE = "./preprocessed_data/{language}_finetuned_teacher_labels/train.csv"'),
        (r'VAL_FILE = ".*preprocessed_data/.*labels/val\.csv"',
         f'VAL_FILE = "./preprocessed_data/{language}_finetuned_teacher_labels/val.csv"'),
        (r'TEST_FILE = ".*preprocessed_data/.*labels/test\.csv"',
         f'TEST_FILE = "./preprocessed_data/{language}_finetuned_teacher_labels/test.csv"'),
        (r'USE_PREFIX = (True|False).*',
         'USE_PREFIX = True  # Match teacher prefix usage'),
    ])
    if not run_command("python train_student_fast.py", f"[{language}] 3/3 Train student"):
        return False

    print(f"\n{'=' * 70}")
    print(f"✓  {language.upper()} COMPLETE")
    print(f"{'=' * 70}")
    return True


def main():
    print("\n" + "=" * 70)
    print("BATCH 2 TRAINING — Ukrainian, Tamil, Telugu, Gujarati, Vietnamese")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    t0 = time.time()
    results = {}

    for language in LANGUAGES:
        success = train_language(language)
        results[language] = "✓ SUCCESS" if success else "✗ FAILED"
        if not success:
            print(f"\nWARNING: {language} failed — continuing with next language...\n")

    hours = (time.time() - t0) / 3600
    print("\n\n" + "=" * 70)
    print("BATCH 2 COMPLETE")
    print(f"Total time : {hours:.2f} hours")
    print(f"End        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    for lang, status in results.items():
        print(f"  {lang:<14} {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()
