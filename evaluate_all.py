"""
Evaluate All 8 Ablation Models — GPU-Crash-Proof Version
==========================================================
Runs full evaluation (ROUGE-1/2/L, BLEU, BERTScore, Semantic Similarity)
on the test set for all existing trained models.

Crash-proofing strategy:
  1. Each model is evaluated in a SUBPROCESS — if GPU crashes, only that
     subprocess dies, not the main orchestrator.
  2. Results for each model are saved to eval_results.json IMMEDIATELY
     after completion. Re-running the script SKIPS already-evaluated models.
  3. BERTScore & Sentence-Transformer run on CPU (no GPU competition).
  4. Generation uses batch_size=1, with OOM fallback to greedy decoding.
  5. Aggressive torch.cuda.empty_cache() + gc.collect() between stages.

Usage:
    python evaluate_all.py                  # evaluate all (skip completed)
    python evaluate_all.py --quick          # 200-sample subset
    python evaluate_all.py --force          # re-evaluate even if results exist
    python evaluate_all.py --only ewad_full # evaluate a single experiment
"""

import os
import sys
import gc
import json
import argparse
import subprocess
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

BASE = os.path.dirname(os.path.abspath(__file__))

# All 8 experiment models with their directories
MODEL_DIRS = {
    "baseline_no_distill":  "student_outputs/baseline_no_distill_20260224_005735",
    "single_teacher_32b":   "student_outputs/single_teacher_32b_20260224_035758",
    "single_teacher_14b":   "student_outputs/single_teacher_14b_20260224_074825",
    "fixed_weights":        "student_outputs/fixed_weights_20260224_114453",
    "confidence_only":      "student_outputs/confidence_only_20260224_163938",
    "agreement_only":       "student_outputs/agreement_only_20260225_103841",
    "ewad_full":            "student_outputs/ewad_full_20260225_180043",
    "ewad_cpdp":            "student_outputs/ewad_cpdp_20260225_235211",
}


def model_already_evaluated(exp_dir):
    """Check if a complete eval_results.json exists for this model."""
    results_file = os.path.join(exp_dir, "eval_results.json")
    if not os.path.exists(results_file):
        return False
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Must have at least ROUGE and semantic similarity to be considered complete
        required_keys = ['rouge1', 'rouge2', 'rougeL', 'semantic_similarity_mean']
        return all(k in data for k in required_keys)
    except (json.JSONDecodeError, KeyError):
        return False


def evaluate_single_model_subprocess(exp_name, best_model, quick=False):
    """
    Run evaluate.py for a single model in a subprocess.
    This isolates GPU crashes — if the subprocess dies, the main process survives.
    Returns True if successful, False otherwise.
    """
    cmd = [
        sys.executable, os.path.join(BASE, "evaluate.py"),
        "--model-dir", best_model,
    ]
    if quick:
        cmd.append("--quick")

    print(f"\n  Launching subprocess: {' '.join(cmd)}")
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=BASE,
            timeout=7200,  # 2-hour timeout per model
            capture_output=False,  # Let output stream to console
        )
        elapsed = time.time() - start_time
        if result.returncode == 0:
            print(f"\n  SUCCESS: {exp_name} evaluated in {elapsed/60:.1f} minutes")
            return True
        else:
            print(f"\n  FAILED: {exp_name} exited with code {result.returncode} after {elapsed/60:.1f} min")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n  TIMEOUT: {exp_name} exceeded 2-hour limit")
        return False
    except Exception as e:
        print(f"\n  ERROR: {exp_name}: {e}")
        return False


def load_results_for_experiment(exp_dir):
    """Load eval_results.json if it exists."""
    results_file = os.path.join(exp_dir, "eval_results.json")
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def print_comparison_table(all_results):
    """Pretty-print a comparison table of all evaluated models."""
    print(f"\n{'='*100}")
    print("COMPARISON TABLE — ALL 8 ABLATIONS")
    print(f"{'='*100}")

    header = f"{'Experiment':<25} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'BLEU-4':>8} {'BERTScr':>8} {'SemSim':>8}"
    print(header)
    print("-" * len(header))

    for exp_name in MODEL_DIRS.keys():
        if exp_name not in all_results:
            print(f"{exp_name:<25} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>8}")
            continue
        r = all_results[exp_name]
        r1 = r.get('rouge1', 0)
        r2 = r.get('rouge2', 0)
        rl = r.get('rougeL', 0)
        bl = r.get('bleu_4', 0)
        bs = r.get('bertscore_f1', 0)
        ss = r.get('semantic_similarity_mean', 0)
        print(f"{exp_name:<25} {r1:>8.4f} {r2:>8.4f} {rl:>8.4f} {bl:>8.4f} {bs:>8.4f} {ss:>8.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate all 8 ablation models (GPU-crash-proof)")
    parser.add_argument("--quick", action="store_true", help="Quick eval on 200 samples")
    parser.add_argument("--force", action="store_true", help="Re-evaluate even if results exist")
    parser.add_argument("--only", type=str, default=None, help="Evaluate only this experiment name")
    args = parser.parse_args()

    experiments_to_run = MODEL_DIRS
    if args.only:
        if args.only not in MODEL_DIRS:
            print(f"Unknown experiment: {args.only}")
            print(f"Available: {', '.join(MODEL_DIRS.keys())}")
            sys.exit(1)
        experiments_to_run = {args.only: MODEL_DIRS[args.only]}

    completed = []
    failed = []
    skipped = []

    for exp_name, rel_dir in experiments_to_run.items():
        exp_dir = os.path.join(BASE, rel_dir)
        best_model = os.path.join(exp_dir, "best_model")
        if not os.path.exists(best_model):
            best_model = os.path.join(exp_dir, "final_model")
        if not os.path.exists(best_model):
            print(f"\n  SKIP {exp_name}: no model found at {exp_dir}")
            skipped.append(exp_name)
            continue

        # Check if already evaluated
        if not args.force and model_already_evaluated(exp_dir):
            print(f"\n  SKIP {exp_name}: already evaluated (use --force to re-run)")
            skipped.append(exp_name)
            continue

        print(f"\n{'#'*80}")
        print(f"# EVALUATING: {exp_name}")
        print(f"# Model: {best_model}")
        print(f"{'#'*80}")

        success = evaluate_single_model_subprocess(exp_name, best_model, quick=args.quick)

        if success:
            completed.append(exp_name)
        else:
            failed.append(exp_name)

        pass  # move on to next model

    # ===== Collect all results (including previously completed) =====
    all_results = {}
    for exp_name, rel_dir in MODEL_DIRS.items():
        exp_dir = os.path.join(BASE, rel_dir)
        r = load_results_for_experiment(exp_dir)
        if r:
            all_results[exp_name] = r

    # Print comparison table
    if all_results:
        print_comparison_table(all_results)

        # Save combined results
        out_file = os.path.join(BASE, "all_eval_results.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nAll results saved to: {out_file}")

    # Status summary
    print(f"\n{'='*100}")
    print("STATUS SUMMARY")
    print(f"{'='*100}")
    print(f"  Completed this run : {len(completed)} — {', '.join(completed) if completed else 'none'}")
    print(f"  Skipped (done/miss): {len(skipped)}  — {', '.join(skipped) if skipped else 'none'}")
    print(f"  Failed             : {len(failed)}   — {', '.join(failed) if failed else 'none'}")
    if failed:
        print(f"\n  To retry failed models, just re-run this script — completed ones will be skipped.")


if __name__ == "__main__":
    main()
