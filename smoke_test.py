"""
Smoke Test: Quick validation using EXISTING Hindi teacher
Tests: Pseudo-label generation → Student training
Should complete in ~3-5 minutes
"""

import os
import sys
import pandas as pd

def main():
    print("\n" + "="*80)
    print("SMOKE TEST: USING EXISTING TEACHER")
    print("="*80)
    
    # Use existing teacher
    teacher_path = "./teachers/hindi_teacher_20260306_181347/final_model"
    if not os.path.exists(teacher_path):
        print(f"ERROR: Teacher not found at {teacher_path}")
        sys.exit(1)
    
    print(f"✓ Using existing teacher: {teacher_path}")
    
    # Create tiny test dataset
    print("\nCreating test dataset (100 samples)...")
    test_dir = "./smoke_test_dir"
    os.makedirs(test_dir, exist_ok=True)
    
    train_df = pd.read_csv("./preprocessed_data/hindi/train.csv").head(100)
    val_df = pd.read_csv("./preprocessed_data/hindi/val.csv").head(20)
    test_df = pd.read_csv("./preprocessed_data/hindi/test.csv").head(20)
    
    train_df.to_csv(f"{test_dir}/train.csv", index=False)
    val_df.to_csv(f"{test_dir}/val.csv", index=False)
    test_df.to_csv(f"{test_dir}/test.csv", index=False)
    
    print(f"✓ Created: 100 train, 20 val, 20 test samples")
    
    # Step 1: Generate pseudo-labels
    print("\n" + "="*80)
    print("STEP 1: GENERATING PSEUDO-LABELS")
    print("="*80)
    
    import generate_teacher_labels
    generate_teacher_labels.TEACHER_MODEL_PATH = teacher_path
    generate_teacher_labels.TRAIN_FILE = f"{test_dir}/train.csv"
    generate_teacher_labels.VAL_FILE = f"{test_dir}/val.csv"
    generate_teacher_labels.TEST_FILE = f"{test_dir}/test.csv"
    generate_teacher_labels.OUTPUT_DIR = f"{test_dir}/labels"
    generate_teacher_labels.BATCH_SIZE = 16
    
    generate_teacher_labels.generate_teacher_predictions()
    print("✓ Pseudo-labels generated")
    
    # Step 2: Train student
    print("\n" + "="*80)
    print("STEP 2: TRAINING STUDENT (1 epoch)")
    print("="*80)
    
    import train_student_fast
    train_student_fast.TRAIN_FILE = f"{test_dir}/labels/train.csv"
    train_student_fast.VAL_FILE = f"{test_dir}/labels/val.csv"
    train_student_fast.TEST_FILE = f"{test_dir}/labels/test.csv"
    train_student_fast.OUTPUT_DIR = f"{test_dir}/student"
    train_student_fast.NUM_EPOCHS = 1
    train_student_fast.EVAL_STEPS = 100
    train_student_fast.SAVE_STEPS = 100
    
    try:
        trainer, test_metrics = train_student_fast.train_student_fast()
        student_rougeL = test_metrics.get('test_rougeL', 0)
        
        print("\n" + "="*80)
        print("✓ SMOKE TEST COMPLETE")
        print("="*80)
        print(f"Student ROUGE-L: {student_rougeL:.4f}")
        print(f"Teacher ROUGE-L was: 0.3715")
        print(f"Retention: {(student_rougeL/0.3715)*100:.1f}%")
        
        if student_rougeL > 0.2:
            print("\n✓ SMOKE TEST PASSED - Pipeline working!")
            print("Ready for full training on all languages")
        else:
            print("\n⚠ SMOKE TEST PARTIAL - Student trained but low scores")
            print("This is expected with tiny dataset. Full data should be better.")
        
    except Exception as e:
        print(f"\n✗ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
