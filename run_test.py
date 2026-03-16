"""
Quick Pipeline Test — 1000 Samples
====================================
Runs the entire pipeline (teacher scoring → student training → evaluation)
with only 1000 samples to verify the architecture works before a full run.

Runs 3 key experiments:
  1. baseline_no_distill  — student alone (sanity check)
  2. ewad_cpdp            — full system (your novel method)
  3. single_teacher_32b   — standard KD baseline for comparison

What to look for:
  - Teacher scoring completes without errors
  - Training loss decreases over steps
  - ewad_cpdp val loss < baseline_no_distill val loss (distillation helps)
  - Evaluation ROUGE scores are non-zero and reasonable

Usage:
    python run_test.py                   # Full test pipeline
    python run_test.py --skip-teachers   # If teacher outputs already exist
    python run_test.py --teachers-only   # Only generate teacher outputs
"""

import os
import sys
import json
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force test mode in config
import config
config.TEST_MODE = True
config.MAX_SAMPLES = 1000
config.STUDENT_NUM_EPOCHS = 1
config.STUDENT_SAVE_STEPS = 50
config.STUDENT_LOGGING_STEPS = 10
config.STUDENT_GRADIENT_ACCUMULATION = 2
config.TEACHER_BATCH_SIZE = 2

# Separate output dirs for test mode (avoids loading 21 GB full-run teacher outputs)
config.TEACHER_32B_OUTPUTS = os.path.join(config.BASE_DIR, "teacher_outputs_test", "teacher_32b")
config.TEACHER_14B_OUTPUTS = os.path.join(config.BASE_DIR, "teacher_outputs_test", "teacher_14b")
config.STUDENT_OUTPUT_DIR = os.path.join(config.BASE_DIR, "student_outputs_test")
config.EVAL_OUTPUT_DIR = os.path.join(config.BASE_DIR, "eval_results_test")

from config import *

# Key experiments to test (covers baseline + best system + simple KD)
TEST_EXPERIMENTS = [
    "baseline_no_distill",
    "ewad_cpdp",
    "single_teacher_32b",
]

LMI_DIR = os.path.dirname(os.path.abspath(__file__))


def run_cmd(cmd, description):
    """Run a command and return success/failure."""
    print(f"\n{'─'*60}")
    print(f"  {description}")
    print(f"  > {cmd}")
    print(f"{'─'*60}\n")
    
    result = subprocess.run(cmd, shell=True, cwd=LMI_DIR)
    
    if result.returncode != 0:
        print(f"\n  FAILED: {description} (exit code {result.returncode})")
        return False
    
    print(f"\n  OK: {description}")
    return True


def test_teacher_generation(teacher="32b"):
    """Generate teacher outputs with --test-mode."""
    return run_cmd(
        f"python generate_teacher_outputs.py --teacher {teacher} --split all --batch-size 2 --test-mode",
        f"Teacher {teacher} scoring (1000 samples)"
    )


def test_student_training(experiment):
    """Train student with --test-mode."""
    return run_cmd(
        f"python train_student.py --experiment {experiment} --test-mode",
        f"Student training: {experiment} (1000 samples, 1 epoch)"
    )


def find_latest_model(experiment_name):
    """Find the most recent output directory for an experiment."""
    if not os.path.exists(STUDENT_OUTPUT_DIR):
        return None
    
    exp_dirs = [
        d for d in os.listdir(STUDENT_OUTPUT_DIR)
        if d.startswith(experiment_name)
    ]
    if not exp_dirs:
        return None
    
    latest = sorted(exp_dirs)[-1]
    return os.path.join(STUDENT_OUTPUT_DIR, latest)


def test_evaluation(experiment_name):
    """Evaluate a trained model."""
    model_dir = find_latest_model(experiment_name)
    if not model_dir:
        print(f"  No model found for {experiment_name}. Skipping evaluation.")
        return False
    
    best_model = os.path.join(model_dir, "best_model")
    if not os.path.exists(best_model):
        best_model = os.path.join(model_dir, "final_model")
    
    if not os.path.exists(best_model):
        print(f"  No best/final model in {model_dir}. Skipping.")
        return False
    
    return run_cmd(
        f'python evaluate.py --model-dir "{best_model}"',
        f"Evaluation: {experiment_name}"
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quick pipeline test with 1000 samples")
    parser.add_argument("--skip-teachers", action="store_true",
                        help="Skip teacher generation (assume outputs exist)")
    parser.add_argument("--teachers-only", action="store_true",
                        help="Only run teacher generation")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation after training")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help=f"Override which experiments to test (default: {TEST_EXPERIMENTS})")
    args = parser.parse_args()

    experiments = args.experiments or TEST_EXPERIMENTS

    print(f"\n{'='*80}")
    print("QUICK PIPELINE TEST — 1000 SAMPLES")
    print(f"{'='*80}")
    print(f"Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Samples:     1000 total -> ~800 train / ~100 val / ~100 test")
    print(f"Epochs:      1")
    print(f"Experiments: {experiments}")
    print(f"{'='*80}\n")

    results = {}

    # ───────────── Phase 1: Teacher scoring ─────────────
    if not args.skip_teachers:
        print(f"\n{'='*80}")
        print("PHASE 1: TEACHER OUTPUT GENERATION (test mode)")
        print(f"{'='*80}")

        for teacher in ["32b", "14b"]:
            ok = test_teacher_generation(teacher)
            results[f"teacher_{teacher}"] = "OK" if ok else "FAILED"
            if not ok:
                print(f"\nTeacher {teacher} failed. You may want to fix this before continuing.")
                # Don't abort — distill experiments will fail but baseline will still work
    else:
        print("\nSkipping teacher generation (--skip-teachers)")

    if args.teachers_only:
        print("\n--teachers-only: stopping here.")
        return

    # ───────────── Phase 2: Student training ─────────────
    print(f"\n{'='*80}")
    print("PHASE 2: STUDENT TRAINING (test mode)")
    print(f"{'='*80}")

    for exp in experiments:
        ok = test_student_training(exp)
        results[f"train_{exp}"] = "OK" if ok else "FAILED"

    # ───────────── Phase 3: Evaluation ─────────────
    if not args.skip_eval:
        print(f"\n{'='*80}")
        print("PHASE 3: EVALUATION (test mode)")
        print(f"{'='*80}")

        for exp in experiments:
            ok = test_evaluation(exp)
            results[f"eval_{exp}"] = "OK" if ok else "FAILED"
    else:
        print("\nSkipping evaluation (--skip-eval)")

    # ───────────── Summary ─────────────
    print(f"\n{'='*80}")
    print("TEST PIPELINE SUMMARY")
    print(f"{'='*80}")
    
    all_ok = True
    for step, status in results.items():
        icon = "PASS" if status == "OK" else "FAIL"
        print(f"  [{icon}] {step}")
        if status != "OK":
            all_ok = False

    if all_ok:
        print(f"\nAll steps passed! Your architecture works.")
        print(f"To run the full pipeline, set TEST_MODE = False in config.py and run:")
        print(f"  python run_all_experiments.py")
    else:
        print(f"\nSome steps failed. Check the errors above before running the full pipeline.")

    # Save test results
    test_results_file = os.path.join(LMI_DIR, "test_pipeline_results.json")
    with open(test_results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "mode": "test (1000 samples)",
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {test_results_file}")


if __name__ == "__main__":
    main()
