"""
Run All Experiments
====================
Orchestrates the full experimental pipeline:
1. Generate teacher outputs (if not already done)
2. Train student for all 8 configurations  
3. Evaluate all models
4. Generate comparison table

Usage:
    python run_all_experiments.py
    python run_all_experiments.py --skip-teachers       # If teacher outputs already exist
    python run_all_experiments.py --experiments ewad_full ewad_cpdp   # Run specific experiments
"""

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

# Import resume helpers from train_student
from train_student import find_latest_experiment_dir, get_completed_epochs


def check_teacher_outputs():
    """Check if teacher outputs exist."""
    results = {}
    for teacher_name, teacher_dir in [("32b", TEACHER_32B_OUTPUTS), ("14b", TEACHER_14B_OUTPUTS)]:
        train_file = os.path.join(teacher_dir, "train.jsonl")
        val_file = os.path.join(teacher_dir, "validation.jsonl")
        test_file = os.path.join(teacher_dir, "test.jsonl")
        
        results[teacher_name] = {
            'train': os.path.exists(train_file),
            'validation': os.path.exists(val_file),
            'test': os.path.exists(test_file),
        }
    return results


def run_teacher_generation():
    """Run teacher output generation for both teachers."""
    print(f"\n{'='*80}")
    print("PHASE 1: TEACHER OUTPUT GENERATION")
    print(f"{'='*80}")
    
    teacher_status = check_teacher_outputs()
    
    for teacher_name in ["32b", "14b"]:
        status = teacher_status[teacher_name]
        missing = [split for split, exists in status.items() if not exists]
        
        if not missing:
            print(f"\n  Teacher {teacher_name}: All outputs exist. Skipping.")
            continue
        
        print(f"\n  Teacher {teacher_name}: Missing splits: {missing}")
        print(f"  Running generation...")
        
        for split in missing:
            cmd = f"python generate_teacher_outputs.py --teacher {teacher_name} --split {split}"
            print(f"  > {cmd}")
            os.system(cmd)


def run_all_training(experiments_to_run=None):
    """Run training for specified experiments with auto-resume support."""
    print(f"\n{'='*80}")
    print("PHASE 2: STUDENT TRAINING (with auto-resume)")
    print(f"{'='*80}")
    
    if experiments_to_run is None:
        experiments_to_run = list(EXPERIMENTS.keys())
    
    trained_models = {}
    
    for i, exp_name in enumerate(experiments_to_run):
        print(f"\n--- Experiment {i+1}/{len(experiments_to_run)}: {exp_name} ---")
        print(f"    {EXPERIMENTS[exp_name]['description']}")
        
        # Check for existing run to auto-resume from
        existing_dir = find_latest_experiment_dir(exp_name)
        if existing_dir:
            completed = get_completed_epochs(existing_dir)
            if completed >= STUDENT_NUM_EPOCHS:
                print(f"  Already completed {completed}/{STUDENT_NUM_EPOCHS} epochs. Skipping training.")
                trained_models[exp_name] = existing_dir
                continue
            else:
                print(f"  Found previous run with {completed}/{STUDENT_NUM_EPOCHS} epochs. Resuming...")
        
        # Use --auto-resume to let train_student.py handle it
        cmd = f"python train_student.py --experiment {exp_name} --auto-resume"
        print(f"  > {cmd}")
        
        ret = os.system(cmd)
        
        if ret != 0:
            print(f"  ERROR: Training failed for {exp_name}")
            # Still try to find the best model from partial training
        
        # Find the latest output directory for this experiment
        latest_dir = find_latest_experiment_dir(exp_name)
        if latest_dir:
            trained_models[exp_name] = latest_dir
            print(f"  Model dir: {latest_dir}")
            
            # === Quick eval after each ablation ===
            best_model = os.path.join(latest_dir, "best_model")
            if not os.path.exists(best_model):
                best_model = os.path.join(latest_dir, "final_model")
            
            if os.path.exists(best_model):
                print(f"  Running quick eval ({QUICK_EVAL_SAMPLES} samples)...")
                eval_cmd = f'python evaluate.py --model-dir "{best_model}" --quick'
                print(f"  > {eval_cmd}")
                os.system(eval_cmd)
    
    # Save model registry
    os.makedirs(STUDENT_OUTPUT_DIR, exist_ok=True)
    registry_file = os.path.join(STUDENT_OUTPUT_DIR, "model_registry.json")
    with open(registry_file, "w") as f:
        json.dump(trained_models, f, indent=2)
    
    return trained_models


def run_all_evaluation(trained_models, quick=False):
    """Evaluate all trained models. quick=True for subset, False for full test set."""
    mode_str = f"QUICK ({QUICK_EVAL_SAMPLES} samples)" if quick else "FULL"
    print(f"\n{'='*80}")
    print(f"PHASE 3: {mode_str} EVALUATION")
    print(f"{'='*80}")
    
    all_results = {}
    
    for exp_name, model_dir in trained_models.items():
        print(f"\nEvaluating: {exp_name}...")
        
        best_model = os.path.join(model_dir, "best_model")
        if not os.path.exists(best_model):
            best_model = os.path.join(model_dir, "final_model")
        
        if not os.path.exists(best_model):
            print(f"  WARNING: No model found at {model_dir}. Skipping.")
            continue
        
        quick_flag = " --quick" if quick else ""
        cmd = f'python evaluate.py --model-dir "{best_model}" --analysis{quick_flag}'
        print(f"  > {cmd}")
        os.system(cmd)
        
        # Load results
        results_file = os.path.join(model_dir, "eval_results.json")
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                all_results[exp_name] = json.load(f)
    
    # Generate comparison table
    if all_results:
        print(f"\n{'='*80}")
        print("COMPARISON TABLE")
        print(f"{'='*80}")
        
        header = f"{'Experiment':<25} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'BLEU-4':>8} {'BERTScr':>8} {'SemSim':>8}"
        print(header)
        print("-" * len(header))
        
        for exp_name, results in all_results.items():
            r1 = results.get('rouge1', 0)
            r2 = results.get('rouge2', 0)
            rl = results.get('rougeL', 0)
            bl = results.get('bleu_4', 0)
            bs = results.get('bertscore_f1', 0)
            ss = results.get('semantic_similarity_mean', 0)
            print(f"{exp_name:<25} {r1:>8.4f} {r2:>8.4f} {rl:>8.4f} {bl:>8.4f} {bs:>8.4f} {ss:>8.4f}")
        
        # Save comparison
        comparison_file = os.path.join(EVAL_OUTPUT_DIR, "comparison_results.json")
        os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
        with open(comparison_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nComparison saved to: {comparison_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run full experimental pipeline")
    parser.add_argument(
        "--skip-teachers",
        action="store_true",
        help="Skip teacher generation (assumes outputs exist)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (only evaluate existing models)"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip final full evaluation (quick eval still runs after each ablation)"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Specific experiments to run (default: all)"
    )
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("DUAL-TEACHER DISTILLATION — FULL PIPELINE")
    print(f"{'='*80}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Teachers: {TEACHER_32B_MODEL} + {TEACHER_14B_MODEL}")
    print(f"Student: {STUDENT_MODEL}")
    print(f"Dataset: BanSum ({DATASET_FILE})")
    
    # Phase 1: Teacher generation
    if not args.skip_teachers:
        run_teacher_generation()
    else:
        print("\n  Skipping teacher generation (--skip-teachers)")
    
    # Phase 2: Training
    if not args.skip_training:
        trained_models = run_all_training(args.experiments)
    else:
        print("\n  Skipping training (--skip-training)")
        # Load existing model registry
        registry_file = os.path.join(STUDENT_OUTPUT_DIR, "model_registry.json")
        if os.path.exists(registry_file):
            with open(registry_file, "r") as f:
                trained_models = json.load(f)
        else:
            print("  ERROR: No model registry found. Run training first.")
            return
    
    # Phase 3: Full evaluation on entire test set
    if not args.skip_eval:
        run_all_evaluation(trained_models, quick=False)
    else:
        print("\n  Skipping final full evaluation (--skip-eval)")
    
    print(f"\n{'='*80}")
    print("ALL DONE!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
