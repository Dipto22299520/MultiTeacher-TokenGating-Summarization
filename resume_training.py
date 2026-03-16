"""
RESUME script — picks up after GPU loss on 2026-03-11.

Already fully complete (skipped):
  - nepali   (teacher + pseudo-labels + student all done)
  - amharic  (teacher + pseudo-labels + student all done)
  - pashto   (teacher + pseudo-labels + student all done)

Remaining work:
  - hausa     : teacher done, pseudo-labels interrupted → regenerate + train student
  - burmese   : nothing done
  - ukrainian : nothing done
  - tamil     : nothing done
  - telugu    : nothing done
  - gujarati  : teacher only had logs (GPU died) → full restart
  - vietnamese: nothing done

Usage:
  python resume_training.py
"""

import glob
import os
import re
import shutil
import subprocess
import time
from datetime import datetime

# ============================================================================
# What remains
# ============================================================================

LANGUAGES = ["tamil", "telugu", "gujarati", "vietnamese"]

# All teachers with final_model are reusable
EXISTING_TEACHERS = {
}

# ============================================================================
# Cleanup incomplete artefacts before starting
# ============================================================================

def cleanup_incomplete():
    """Remove artefacts that are partially written and would confuse the pipeline."""
    import glob as _glob

    # Remove any teacher dir that has no final_model (GPU crash leftovers)
    for teacher_dir in _glob.glob("./teachers/*_teacher_*"):
        final = os.path.join(teacher_dir, "final_model")
        if not os.path.exists(final):
            print(f"Removing incomplete teacher dir: {teacher_dir}")
            shutil.rmtree(teacher_dir)

    print("Cleanup done.\n")

# ============================================================================
# Helpers  (identical to train_new_languages.py)
# ============================================================================

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
        raise FileNotFoundError(f"No teacher found for '{language}' matching: {pattern}")
    return max(matches, key=os.path.getmtime)


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

# ============================================================================
# Per-language pipeline
# ============================================================================

def train_language(language):
    print(f"\n\n{'#' * 80}")
    print(f"# STARTING: {language.upper()}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 80}")

    for f in [
        f"./preprocessed_data/{language}/train.csv",
        f"./preprocessed_data/{language}/val.csv",
        f"./preprocessed_data/{language}/test.csv",
    ]:
        if not os.path.exists(f):
            print(f"ERROR: Missing {f}")
            print(f"  Run preprocess_new_languages.py or preprocess_batch2.py first.")
            return False

    # Step 1: teacher
    teacher_model_path = EXISTING_TEACHERS.get(language)
    if teacher_model_path:
        if not os.path.exists(teacher_model_path):
            print(f"ERROR: Existing teacher not found: {teacher_model_path}")
            return False
        print(f"Reusing existing teacher: {teacher_model_path}")
    else:
        update_config_file("train_teacher.py", [
            (r'TRAIN_FILE = ".*"',  f'TRAIN_FILE = "./preprocessed_data/{language}/train.csv"'),
            (r'VAL_FILE = ".*"',    f'VAL_FILE = "./preprocessed_data/{language}/val.csv"'),
            (r'TEST_FILE = ".*"',   f'TEST_FILE = "./preprocessed_data/{language}/test.csv"'),
            (r'OUTPUT_DIR = ".*"',  f'OUTPUT_DIR = "./teachers/{language}_teacher"'),
        ])
        if not run_command("python train_teacher.py", f"[{language}] Step 1/3: Train teacher"):
            return False
        try:
            teacher_model_path = find_latest_teacher(language)
            print(f"Teacher: {teacher_model_path}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return False

    teacher_model_path = teacher_model_path.replace("\\", "/")

    # Step 2: pseudo-labels
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
    if not run_command("python generate_teacher_labels.py",
                       f"[{language}] Step 2/3: Generate pseudo-labels"):
        return False

    # Step 3: student
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
    if not run_command("python train_student_fast.py",
                       f"[{language}] Step 3/3: Train student"):
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
    print("RESUME TRAINING — 6 REMAINING LANGUAGES")
    print("=" * 80)
    print(f"Skipped (done): nepali, amharic, pashto, hausa")
    print(f"Remaining     : {', '.join(l.capitalize() for l in LANGUAGES)}")
    print(f"Start         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    cleanup_incomplete()

    t0 = time.time()
    results = {}

    for language in LANGUAGES:
        success = train_language(language)
        results[language] = "✓ SUCCESS" if success else "✗ FAILED"
        if not success:
            print(f"\nWARNING: {language} failed — continuing with next...\n")

    hours = (time.time() - t0) / 3600
    print("\n\n" + "=" * 80)
    print("RESUME COMPLETE — SUMMARY")
    print("=" * 80)
    print(f"Total time : {hours:.2f} hours")
    print(f"Ended      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("  nepali         ✓ (was already done)")
    print("  amharic        ✓ (was already done)")
    print("  pashto         ✓ (was already done)")
    print("  hausa          ✓ (was already done)")
    for lang, status in results.items():
        print(f"  {lang:<14} {status}")
    print("=" * 80)


if __name__ == "__main__":
    main()
