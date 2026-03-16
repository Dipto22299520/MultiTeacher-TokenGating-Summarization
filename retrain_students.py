"""
Retrain ALL students only — teachers and pseudo-labels already exist.
Uses the fixed hyperparameters (more epochs, smaller batch, proper warmup).
"""

import glob
import os
import re
import subprocess
import time
from datetime import datetime

# ============================================================================
# All 10 languages — pseudo-labels must already exist for each
# ============================================================================

LANGUAGES = [
    "nepali", "amharic", "pashto", "hausa", "burmese",
    "ukrainian", "tamil", "telugu", "gujarati", "vietnamese",
]

def update_config_file(filepath, updates):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    for pattern, replacement in updates:
        content = re.sub(pattern, lambda _m, r=replacement: r, content, flags=re.MULTILINE)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def run_command(cmd, description):
    print(f"\n{'=' * 80}")
    print(description)
    print(f"{'=' * 80}")
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


def train_student(language):
    print(f"\n\n{'#' * 80}")
    print(f"# STUDENT TRAINING: {language.upper()}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 80}")

    labels_dir = f"./preprocessed_data/{language}_finetuned_teacher_labels"
    for fname in ["train.csv", "val.csv", "test.csv"]:
        fpath = os.path.join(labels_dir, fname)
        if not os.path.exists(fpath):
            print(f"ERROR: Missing {fpath}")
            print(f"  Pseudo-labels not found — run teacher + generate_teacher_labels first.")
            return False

    update_config_file("train_student_fast.py", [
        (r'OUTPUT_DIR = ".*students.*"',
         f'OUTPUT_DIR = "./students/{language}_student_fast"'),
        (r'TRAIN_FILE = ".*preprocessed_data/.*labels/train\.csv"',
         f'TRAIN_FILE = "./{labels_dir}/train.csv"'),
        (r'VAL_FILE = ".*preprocessed_data/.*labels/val\.csv"',
         f'VAL_FILE = "./{labels_dir}/val.csv"'),
        (r'TEST_FILE = ".*preprocessed_data/.*labels/test\.csv"',
         f'TEST_FILE = "./{labels_dir}/test.csv"'),
        (r'USE_PREFIX = (True|False).*',
         'USE_PREFIX = True  # Match teacher prefix usage'),
    ])

    if not run_command("python train_student_fast.py",
                       f"[{language}] Train student"):
        return False

    print(f"\n✓  {language.upper()} student done")
    return True


def main():
    print("\n" + "=" * 80)
    print("RETRAIN ALL STUDENTS (teachers + pseudo-labels already exist)")
    print("=" * 80)
    print(f"Languages : {', '.join(l.capitalize() for l in LANGUAGES)}")
    print(f"Start     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    t0 = time.time()
    results = {}

    for language in LANGUAGES:
        success = train_student(language)
        results[language] = "✓ SUCCESS" if success else "✗ FAILED"
        if not success:
            print(f"\nWARNING: {language} student failed — continuing...\n")

    hours = (time.time() - t0) / 3600
    print("\n\n" + "=" * 80)
    print("ALL STUDENTS RETRAINED — SUMMARY")
    print("=" * 80)
    print(f"Total time : {hours:.2f} hours")
    print(f"Ended      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    for lang, status in results.items():
        print(f"  {lang:<14} {status}")
    print("=" * 80)


if __name__ == "__main__":
    main()
