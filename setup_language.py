"""
Helper script to configure training for a specific language.
Usage: python setup_language.py <language>
Example: python setup_language.py hindi
"""

import sys
import os

def setup_language(language):
    """Configure train_teacher.py and train_student.py for a specific language."""
    
    # Validate language
    valid_languages = ["hindi", "urdu", "gujarati", "nepali", "sinhala"]
    if language.lower() not in valid_languages:
        print(f"Error: Invalid language '{language}'")
        print(f"Valid languages: {', '.join(valid_languages)}")
        return False
    
    language = language.lower()
    
    print(f"\n{'='*80}")
    print(f"CONFIGURING TRAINING FOR: {language.upper()}")
    print(f"{'='*80}\n")
    
    # Paths
    lang_dir = f"./preprocessed_data/{language}"
    train_file = f"{lang_dir}/train.csv"
    val_file = f"{lang_dir}/val.csv"
    test_file = f"{lang_dir}/test.csv"
    
    # Check if files exist
    for f in [train_file, val_file, test_file]:
        if not os.path.exists(f):
            print(f"Error: {f} does not exist")
            return False
    
    print("✓ Dataset files found:")
    print(f"  - {train_file}")
    print(f"  - {val_file}")
    print(f"  - {test_file}")
    
    # Read train_teacher.py
    with open("train_teacher.py", "r", encoding="utf-8") as f:
        teacher_content = f.read()
    
    # Update teacher paths
    teacher_content = teacher_content.replace(
        'TRAIN_FILE = "train.csv"',
        f'TRAIN_FILE = "{train_file}"'
    ).replace(
        'VAL_FILE = "val.csv"',
        f'VAL_FILE = "{val_file}"'
    ).replace(
        'TEST_FILE = "test.csv"',
        f'TEST_FILE = "{test_file}"'
    ).replace(
        'OUTPUT_DIR = "./mt5_xlsum_teacher_finetuned"',
        f'OUTPUT_DIR = "./teachers/{language}_teacher"'
    )
    
    # Save updated teacher script
    with open("train_teacher.py", "w", encoding="utf-8") as f:
        f.write(teacher_content)
    
    print(f"\n✓ Updated train_teacher.py for {language}")
    print(f"  Output will be saved to: ./teachers/{language}_teacher_<timestamp>/")
    
    # Read train_student.py
    with open("train_student.py", "r", encoding="utf-8") as f:
        student_content = f.read()
    
    # Update student paths
    student_content = student_content.replace(
        'TRAIN_FILE = "train.csv"',
        f'TRAIN_FILE = "{train_file}"'
    ).replace(
        'VAL_FILE = "val.csv"',
        f'VAL_FILE = "{val_file}"'
    ).replace(
        'TEST_FILE = "test.csv"',
        f'TEST_FILE = "{test_file}"'
    ).replace(
        'OUTPUT_DIR = "./mt5_small_student_distilled"',
        f'OUTPUT_DIR = "./students/{language}_student"'
    )
    
    # Save updated student script
    with open("train_student.py", "w", encoding="utf-8") as f:
        f.write(student_content)
    
    print(f"✓ Updated train_student.py for {language}")
    print(f"  Output will be saved to: ./students/{language}_student_<timestamp>/")
    
    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print(f"{'='*80}")
    print(f"\n1. Train teacher model:")
    print(f"   python train_teacher.py")
    print(f"\n2. After teacher training completes, update TEACHER_MODEL_PATH in train_student.py")
    print(f"   (Look for the timestamped directory under ./teachers/{language}_teacher_*/final_model)")
    print(f"\n3. Train student model:")
    print(f"   python train_student.py")
    print(f"\n{'='*80}\n")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python setup_language.py <language>")
        print("Available languages: hindi, urdu, gujarati, nepali, sinhala")
        sys.exit(1)
    
    language = sys.argv[1]
    success = setup_language(language)
    
    if not success:
        sys.exit(1)
