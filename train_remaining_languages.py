"""
Automated training pipeline for remaining 4 languages
Trains teacher -> generates pseudo-labels -> trains student
Runs sequentially: Urdu, Gujarati, Nepali, Sinhala
"""

import glob
import os
import re
import subprocess
import time
from datetime import datetime

LANGUAGES = ["amharic", "hausa", "persian", "nepali", "pashto"]

# Optional: use already-trained teachers to skip retraining
EXISTING_TEACHERS = {
}


def update_config_file(filepath, updates):
    """Update configuration variables in a Python file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    for pattern, replacement in updates:
        # Use a lambda to avoid backslash escapes in replacement paths
        content = re.sub(pattern, lambda _m, r=replacement: r, content, flags=re.MULTILINE)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def find_latest_teacher(language):
    """Find the most recently created teacher model directory."""
    pattern = f"./teachers/{language}_teacher_*/final_model"
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No teacher model found for {language}")
    return max(matches, key=os.path.getmtime)


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'=' * 80}")
    print(description)
    print(f"{'=' * 80}")
    print(f"Command: {cmd}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")

    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    elapsed = time.time() - start_time

    print(f"Completed in {elapsed / 60:.1f} minutes")
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        return False
    return True


def train_language(language):
    """Train teacher and student for one language."""
    print(f"\n\n{'#' * 80}")
    print(f"# STARTING {language.upper()}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 80}\n")

    teacher_model_path = EXISTING_TEACHERS.get(language)
    if teacher_model_path:
        if not os.path.exists(teacher_model_path):
            print(f"ERROR: Existing teacher not found at {teacher_model_path}")
            return False
        print(f"Using existing teacher for {language}: {teacher_model_path}")
    else:
        # Step 1: Update train_teacher.py for this language
        print(f"Configuring train_teacher.py for {language}...")
        update_config_file(
            "train_teacher.py",
            [
                (r'TRAIN_FILE = ".*"', f'TRAIN_FILE = "./preprocessed_data/{language}/train.csv"'),
                (r'VAL_FILE = ".*"', f'VAL_FILE = "./preprocessed_data/{language}/val.csv"'),
                (r'TEST_FILE = ".*"', f'TEST_FILE = "./preprocessed_data/{language}/test.csv"'),
                (r'OUTPUT_DIR = ".*"', f'OUTPUT_DIR = "./teachers/{language}_teacher"'),
            ],
        )

        # Step 2: Train teacher
        if not run_command("python train_teacher.py", f"Training {language} teacher model"):
            return False

        # Step 3: Find the trained teacher model
        print(f"\nFinding trained {language} teacher model...")
        try:
            teacher_model_path = find_latest_teacher(language)
            print(f"✓ Found teacher: {teacher_model_path}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return False

    # Normalize teacher path for consistency (Windows backslashes can break configs)
    teacher_model_path = teacher_model_path.replace("\\", "/")

    # Step 4: Update generate_teacher_labels.py to use the trained teacher
    print(f"\nConfiguring generate_teacher_labels.py for {language}...")
    update_config_file(
        "generate_teacher_labels.py",
        [
            (r'TEACHER_MODEL_PATH = ".*"', f'TEACHER_MODEL_PATH = "{teacher_model_path}"'),
            (r'TRAIN_FILE = ".*preprocessed_data/[^/]+/train\.csv"', f'TRAIN_FILE = "./preprocessed_data/{language}/train.csv"'),
            (r'VAL_FILE = ".*preprocessed_data/[^/]+/val\.csv"', f'VAL_FILE = "./preprocessed_data/{language}/val.csv"'),
            (r'TEST_FILE = ".*preprocessed_data/[^/]+/test\.csv"', f'TEST_FILE = "./preprocessed_data/{language}/test.csv"'),
            (r'OUTPUT_DIR = ".*preprocessed_data/.*"', f'OUTPUT_DIR = "./preprocessed_data/{language}_finetuned_teacher_labels"'),
            (r'USE_PREFIX = (True|False).*', 'USE_PREFIX = True  # mT5 teachers were trained with prefix'),
        ],
    )

    # Step 5: Generate pseudo-labels
    if not run_command("python generate_teacher_labels.py", f"Generating pseudo-labels for {language}"):
        return False

    # Step 6: Update train_student_fast.py to use generated labels
    print(f"\nConfiguring train_student_fast.py for {language}...")
    update_config_file(
        "train_student_fast.py",
        [
            (r'OUTPUT_DIR = ".*students.*"', f'OUTPUT_DIR = "./students/{language}_student_fast"'),
            (r'TRAIN_FILE = ".*preprocessed_data/.*labels/train\.csv"', f'TRAIN_FILE = "./preprocessed_data/{language}_finetuned_teacher_labels/train.csv"'),
            (r'VAL_FILE = ".*preprocessed_data/.*labels/val\.csv"', f'VAL_FILE = "./preprocessed_data/{language}_finetuned_teacher_labels/val.csv"'),
            (r'TEST_FILE = ".*preprocessed_data/.*labels/test\.csv"', f'TEST_FILE = "./preprocessed_data/{language}_finetuned_teacher_labels/test.csv"'),
            (r'USE_PREFIX = (True|False).*', 'USE_PREFIX = True  # Match teacher prefix usage'),
        ],
    )

    # Step 7: Train student
    if not run_command("python train_student_fast.py", f"Training {language} student model"):
        return False

    print(f"\n{'=' * 80}")
    print(f"✓ {language.upper()} COMPLETE")
    print(f"{'=' * 80}\n")
    return True


def main():
    """Run full training pipeline."""
    print("\n" + "=" * 80)
    print("AUTOMATED TRAINING: 4 LANGUAGES")
    print("=" * 80)
    print(f"Languages: {', '.join([l.capitalize() for l in LANGUAGES])}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    overall_start = time.time()
    results = {}

    for language in LANGUAGES:
        success = train_language(language)
        results[language] = "✓ SUCCESS" if success else "✗ FAILED"

        if not success:
            print(f"\n{'!' * 80}")
            print(f"WARNING: {language} failed, continuing with next language...")
            print(f"{'!' * 80}\n")

    overall_time = (time.time() - overall_start) / 3600

    print("\n\n" + "=" * 80)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"Total time: {overall_time:.2f} hours")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults:")
    for lang, status in results.items():
        print(f"  {lang.capitalize():12s} {status}")
    print("=" * 80)


if __name__ == "__main__":
    main()
