"""
Automated training pipeline for all 10 new languages:

  Batch 1 (low-resource):
    - Nepali   (Devanagari, South Asia)
    - Amharic  (Ethiopic, East Africa)
    - Pashto   (Arabic-variant, Afghanistan/Pakistan)
    - Hausa    (Latin, West Africa)
    - Burmese  (Myanmar, Southeast Asia)

  Batch 2:
    - Ukrainian
    - Tamil
    - Telugu
    - Gujarati
    - Vietnamese

For each language the pipeline runs:
  1. Fine-tune teacher  (csebuetnlp/mT5_multilingual_XLSum)
  2. Generate pseudo-labels from teacher
  3. Train student      (google/mt5-small) on pseudo-labels

Prerequisites:
  - python preprocess_new_languages.py   (batch 1 data)
  - python preprocess_batch2.py          (batch 2 data)

Usage:
  python train_new_languages.py
"""

import glob
import os
import re
import subprocess
import time
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

LANGUAGES = [
    # Batch 1 — low-resource
    "nepali", "amharic", "pashto", "hausa", "burmese",
    # Batch 2
    "ukrainian", "tamil", "telugu", "gujarati", "vietnamese",
]

# If you already have a trained teacher for a language, add its path here
# to skip teacher training and go straight to pseudo-label generation.
# Example:
#   EXISTING_TEACHERS = {
#       "nepali": "./teachers/nepali_teacher_20260310_120000/final_model",
#   }
EXISTING_TEACHERS = {}

# ============================================================================
# Helpers
# ============================================================================

def update_config_file(filepath, updates):
    """Update configuration variables in a Python file using regex substitution."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    for pattern, replacement in updates:
        content = re.sub(pattern, lambda _m, r=replacement: r, content, flags=re.MULTILINE)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def find_latest_teacher(language):
    """Return path to the most-recently created teacher final_model directory."""
    pattern = f"./teachers/{language}_teacher_*/final_model"
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No teacher model found for '{language}' matching pattern: {pattern}"
        )
    return max(matches, key=os.path.getmtime)


def run_command(cmd, description):
    """Run a shell command, print status, and return success flag."""
    print(f"\n{'=' * 80}")
    print(description)
    print(f"{'=' * 80}")
    print(f"Command : {cmd}")
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    elapsed = time.time() - start_time

    print(f"Finished in {elapsed / 60:.1f} min | return code: {result.returncode}")
    if result.returncode != 0:
        print(f"ERROR: '{cmd}' failed (return code {result.returncode})")
        return False
    return True


# ============================================================================
# Per-language pipeline
# ============================================================================

def train_language(language):
    """Run the full teacher → pseudo-labels → student pipeline for one language."""
    print(f"\n\n{'#' * 80}")
    print(f"# STARTING: {language.upper()}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 80}")

    # Verify preprocessed data exists
    required_files = [
        f"./preprocessed_data/{language}/train.csv",
        f"./preprocessed_data/{language}/val.csv",
        f"./preprocessed_data/{language}/test.csv",
    ]
    for f in required_files:
        if not os.path.exists(f):
            print(f"ERROR: Missing {f}")
            print(f"  Run: python preprocess_new_languages.py  (batch 1)")
            print(f"  Run: python preprocess_batch2.py         (batch 2)")
            return False

    # ------------------------------------------------------------------
    # Step 1 (optional): use an existing teacher or train a new one
    # ------------------------------------------------------------------
    teacher_model_path = EXISTING_TEACHERS.get(language)

    if teacher_model_path:
        if not os.path.exists(teacher_model_path):
            print(f"ERROR: Existing teacher path not found: {teacher_model_path}")
            return False
        print(f"Using pre-existing teacher: {teacher_model_path}")
    else:
        print(f"\nConfiguring train_teacher.py for {language}...")
        update_config_file(
            "train_teacher.py",
            [
                (r'TRAIN_FILE = ".*"',  f'TRAIN_FILE = "./preprocessed_data/{language}/train.csv"'),
                (r'VAL_FILE = ".*"',    f'VAL_FILE = "./preprocessed_data/{language}/val.csv"'),
                (r'TEST_FILE = ".*"',   f'TEST_FILE = "./preprocessed_data/{language}/test.csv"'),
                (r'OUTPUT_DIR = ".*"',  f'OUTPUT_DIR = "./teachers/{language}_teacher"'),
            ],
        )

        if not run_command("python train_teacher.py", f"[{language}] Step 1/3: Train teacher"):
            return False

        try:
            teacher_model_path = find_latest_teacher(language)
            print(f"Teacher model: {teacher_model_path}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return False

    # Normalise path separators (important on Windows)
    teacher_model_path = teacher_model_path.replace("\\", "/")

    # ------------------------------------------------------------------
    # Step 2: Generate pseudo-labels
    # ------------------------------------------------------------------
    print(f"\nConfiguring generate_teacher_labels.py for {language}...")
    update_config_file(
        "generate_teacher_labels.py",
        [
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
        ],
    )

    if not run_command(
        "python generate_teacher_labels.py",
        f"[{language}] Step 2/3: Generate pseudo-labels",
    ):
        return False

    # ------------------------------------------------------------------
    # Step 3: Train student on cached pseudo-labels
    # ------------------------------------------------------------------
    print(f"\nConfiguring train_student_fast.py for {language}...")
    update_config_file(
        "train_student_fast.py",
        [
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
        ],
    )

    if not run_command(
        "python train_student_fast.py",
        f"[{language}] Step 3/3: Train student",
    ):
        return False

    print(f"\n{'=' * 80}")
    print(f"✓  {language.upper()} COMPLETE")
    print(f"{'=' * 80}")
    return True


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("AUTOMATED TRAINING PIPELINE — 10 NEW LANGUAGES")
    print("=" * 80)
    print(f"Languages : {', '.join(l.capitalize() for l in LANGUAGES)}")
    print(f"Start     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    overall_start = time.time()
    results = {}

    for language in LANGUAGES:
        success = train_language(language)
        results[language] = "✓ SUCCESS" if success else "✗ FAILED"

        if not success:
            print(f"\n{'!' * 80}")
            print(f"WARNING: {language} failed — continuing with next language...")
            print(f"{'!' * 80}")

    total_hours = (time.time() - overall_start) / 3600

    print("\n\n" + "=" * 80)
    print("PIPELINE COMPLETE — ALL 10 LANGUAGES SUMMARY")
    print("=" * 80)
    print(f"Total time : {total_hours:.2f} hours")
    print(f"Ended      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults:")
    for lang, status in results.items():
        print(f"  {lang.capitalize():12s}  {status}")
    print("=" * 80)
    print("\nNext step: update and run evaluate_all_models.py with the new model paths.")
    print("=" * 80)


if __name__ == "__main__":
    main()
