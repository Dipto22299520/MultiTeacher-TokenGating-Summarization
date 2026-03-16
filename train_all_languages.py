"""
Automated training pipeline for all 5 languages:
Fine-tune mT5-XLSum teacher → Distill to mT5-small student
"""

import os
import sys
import subprocess
import json
from datetime import datetime

# Languages to process
LANGUAGES = ["hindi", "urdu", "gujarati", "nepali", "sinhala"]

# Base directories
PREPROCESSED_DIR = "./preprocessed_data"
TEACHER_OUTPUT_BASE = "./teachers"
STUDENT_OUTPUT_BASE = "./students"

# Create output directories
os.makedirs(TEACHER_OUTPUT_BASE, exist_ok=True)
os.makedirs(STUDENT_OUTPUT_BASE, exist_ok=True)

def update_teacher_config(language):
    """Update train_teacher.py paths for specific language."""
    # Use forward slashes (work on Windows too)
    lang_dir = f"./preprocessed_data/{language}"
    
    # Read the teacher script
    with open("train_teacher.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Update file paths
    content = content.replace(
        'TRAIN_FILE = "train.csv"',
        f'TRAIN_FILE = "{lang_dir}/train.csv"'
    )
    content = content.replace(
        'VAL_FILE = "val.csv"',
        f'VAL_FILE = "{lang_dir}/val.csv"'
    )
    content = content.replace(
        'TEST_FILE = "test.csv"',
        f'TEST_FILE = "{lang_dir}/test.csv"'
    )
    content = content.replace(
        'OUTPUT_DIR = "./mt5_xlsum_teacher_finetuned"',
        f'OUTPUT_DIR = "./teachers/{language}_teacher"'
    )
    
    # Write to temporary file
    temp_file = f"train_teacher_{language}.py"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    return temp_file

def update_student_config(language, teacher_path):
    """Update train_student.py paths for specific language."""
    # Use forward slashes (work on Windows too)
    lang_dir = f"./preprocessed_data/{language}"
    
    # Read the student script
    with open("train_student.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Update file paths
    content = content.replace(
        'TEACHER_MODEL_PATH = "./mt5_xlsum_teacher_finetuned/final_model"',
        f'TEACHER_MODEL_PATH = "{teacher_path}"'
    )
    content = content.replace(
        'TRAIN_FILE = "train.csv"',
        f'TRAIN_FILE = "{lang_dir}/train.csv"'
    )
    content = content.replace(
        'VAL_FILE = "val.csv"',
        f'VAL_FILE = "{lang_dir}/val.csv"'
    )
    content = content.replace(
        'TEST_FILE = "test.csv"',
        f'TEST_FILE = "{lang_dir}/test.csv"'
    )
    content = content.replace(
        'OUTPUT_DIR = "./mt5_small_student_distilled"',
        f'OUTPUT_DIR = "./students/{language}_student"'
    )
    
    # Write to temporary file
    temp_file = f"train_student_{language}.py"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    return temp_file

def train_language(language):
    """Train teacher and student for a specific language."""
    print("\n" + "="*80)
    print(f"PROCESSING LANGUAGE: {language.upper()}")
    print("="*80)
    
    # Step 1: Train teacher
    print(f"\n{'='*80}")
    print(f"STEP 1: Training Teacher Model for {language}")
    print(f"{'='*80}\n")
    
    teacher_script = update_teacher_config(language)
    
    try:
        result = subprocess.run(
            ["python", teacher_script],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ Teacher training completed for {language}")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Teacher training failed for {language}")
        print(f"Error: {e}")
        return False
    finally:
        # Clean up temp file
        if os.path.exists(teacher_script):
            os.remove(teacher_script)
    
    # Find the teacher model directory (with timestamp)
    teacher_base = f"./teachers/{language}_teacher"
    
    # List all directories in teachers folder
    if not os.path.exists("./teachers"):
        print(f"\n✗ Teachers directory not found")
        return False
    
    teacher_dirs = [d for d in os.listdir("./teachers") if d.startswith(f"{language}_teacher_")]
    
    if not teacher_dirs:
        print(f"\n✗ Could not find trained teacher model for {language}")
        return False
    
    # Get the most recent teacher directory
    teacher_dirs.sort(reverse=True)
    teacher_path = f"./teachers/{teacher_dirs[0]}/final_model"
    
    # Step 2: Train student
    print(f"\n{'='*80}")
    print(f"STEP 2: Training Student Model for {language}")
    print(f"{'='*80}\n")
    
    student_script = update_student_config(language, teacher_path)
    
    try:
        result = subprocess.run(
            ["python", student_script],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ Student training completed for {language}")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Student training failed for {language}")
        print(f"Error: {e}")
        return False
    finally:
        # Clean up temp file
        if os.path.exists(student_script):
            os.remove(student_script)
    
    print(f"\n{'='*80}")
    print(f"✓ COMPLETED: {language.upper()}")
    print(f"{'='*80}")
    
    return True

def main():
    """Main pipeline - train all languages."""
    print(f"\n{'='*80}")
    print("MULTILINGUAL TEACHER-STUDENT TRAINING PIPELINE")
    print(f"{'='*80}")
    print(f"\nLanguages: {', '.join(LANGUAGES)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    for i, language in enumerate(LANGUAGES, 1):
        print(f"\n\n{'#'*80}")
        print(f"# LANGUAGE {i}/{len(LANGUAGES)}: {language.upper()}")
        print(f"{'#'*80}")
        
        success = train_language(language)
        results[language] = "SUCCESS" if success else "FAILED"
        
        if not success:
            print(f"\n⚠ Warning: {language} training failed. Continuing with next language...")
    
    # Summary
    print(f"\n\n{'='*80}")
    print("TRAINING PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults:")
    for lang, status in results.items():
        symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"  {symbol} {lang}: {status}")
    
    # Save results
    results_file = "training_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
