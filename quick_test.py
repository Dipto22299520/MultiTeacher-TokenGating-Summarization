"""
Quick smoke test: Train tiny teacher WITH prefix, then student
Validates the entire pipeline works correctly
"""

import os
import sys
import pandas as pd

def main():
    print("\n" + "="*80)
    print("SMOKE TEST: TEACHER + STUDENT WITH PREFIX")
    print("="*80)
    
    # Create tiny dataset (100 samples)
    print("\nCreating tiny dataset...")
    test_dir = "./quick_test"
    os.makedirs(test_dir, exist_ok=True)
    
    # Load and sample
    train_df = pd.read_csv("./preprocessed_data/hindi/train.csv").head(100)
    val_df = pd.read_csv("./preprocessed_data/hindi/val.csv").head(20)
    test_df = pd.read_csv("./preprocessed_data/hindi/test.csv").head(20)
    
    train_df.to_csv(f"{test_dir}/train.csv", index=False)
    val_df.to_csv(f"{test_dir}/val.csv", index=False)
    test_df.to_csv(f"{test_dir}/test.csv", index=False)
    
    print(f"✓ Dataset: 100 train, 20 val, 20 test")
    
    # Step 1: Train teacher
    print("\n" + "="*80)
    print("STEP 1: TRAINING TEACHER (1 epoch, 100 samples)")
    print("="*80)
    
    import train_teacher
    train_teacher.TRAIN_FILE = f"{test_dir}/train.csv"
    train_teacher.VAL_FILE = f"{test_dir}/val.csv"
    train_teacher.TEST_FILE = f"{test_dir}/test.csv"
    train_teacher.OUTPUT_DIR = f"{test_dir}/teacher"
    train_teacher.NUM_EPOCHS = 1
    train_teacher.EVAL_STEPS = 50
    train_teacher.SAVE_STEPS = 50
    
    trainer, test_metrics = train_teacher.train_model()
    teacher_rougeL = test_metrics.get('test_rougeL', 0)
    print(f"\n✓ Teacher ROUGE-L: {teacher_rougeL:.4f}")
    
    # Find teacher path
    teacher_dirs = [d for d in os.listdir(test_dir) if d.startswith("teacher_")]
    teacher_dirs.sort(reverse=True)
    teacher_path = f"{test_dir}/{teacher_dirs[0]}/final_model"
    
    # Step 2: Generate pseudo-labels
    print("\n" + "="*80)
    print("STEP 2: GENERATING PSEUDO-LABELS")
    print("="*80)
    
    import generate_teacher_labels
    generate_teacher_labels.TEACHER_MODEL_PATH = teacher_path
    generate_teacher_labels.TRAIN_FILE = f"{test_dir}/train.csv"
    generate_teacher_labels.VAL_FILE = f"{test_dir}/val.csv"
    generate_teacher_labels.TEST_FILE = f"{test_dir}/test.csv"
    generate_teacher_labels.OUTPUT_DIR = f"{test_dir}/data_with_labels"
    generate_teacher_labels.BATCH_SIZE = 8
    
    generate_teacher_labels.generate_teacher_predictions()
    print("✓ Pseudo-labels generated")
    
    # Step 3: Train student
    print("\n" + "="*80)
    print("STEP 3: TRAINING STUDENT (1 epoch)")
    print("="*80)
    
    import train_student_fast
    train_student_fast.TRAIN_FILE = f"{test_dir}/data_with_labels/train.csv"
    train_student_fast.VAL_FILE = f"{test_dir}/data_with_labels/val.csv"
    train_student_fast.TEST_FILE = f"{test_dir}/data_with_labels/test.csv"
    train_student_fast.OUTPUT_DIR = f"{test_dir}/student"
    train_student_fast.NUM_EPOCHS = 1
    train_student_fast.EVAL_STEPS = 50
    train_student_fast.SAVE_STEPS = 50
    
    trainer, test_metrics = train_student_fast.train_student_fast()
    student_rougeL = test_metrics.get('test_rougeL', 0)
    
    print("\n" + "="*80)
    print("SMOKE TEST RESULTS")
    print("="*80)
    print(f"Teacher ROUGE-L: {teacher_rougeL:.4f}")
    print(f"Student ROUGE-L: {student_rougeL:.4f}")
    
    # Validation
    if teacher_rougeL > 0.15 and student_rougeL > 0.1:
        print("\n✓ SMOKE TEST PASSED!")
        print("Both teacher and student are learning (scores > baseline)")
        print("Ready for full training!")
        return True
    else:
        print("\n✗ SMOKE TEST FAILED")
        print(f"Scores too low - models not learning properly")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
