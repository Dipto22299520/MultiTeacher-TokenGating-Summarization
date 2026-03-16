"""
Ablation Study Runner (BanSum Version)
=======================================
Orchestrates the full ablation study on bansum_lte_1000 dataset:
1. Ensures pseudo-labels are generated (from BanSum-finetuned mT5 teachers)
2. Runs all 5 ablation configurations sequentially
3. Collects results and prints a comparison table

Usage:
  python run_ablation_bansum.py           # Full ablation (all 5 configs)
  python run_ablation_bansum.py --quick   # Quick test mode
  python run_ablation_bansum.py --configs A1_baseline A5_full_pipeline  # Subset
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime

CONFIGS = [
    "A1_baseline",
    "A2_single_kd",
    "A3_multi_teacher",
    "A4_adaptive_temp",
    "A5_full_pipeline",
]

RESULTS_DIR = "ablation_results_bansum"
PSEUDO_LABEL_DIR = "data/pseudo_labels_bansum"


def ensure_pseudo_labels(quick=False):
    """Generate pseudo-labels from BanSum-finetuned mT5 teachers if they don't exist."""
    needed = any(
        not os.path.exists(os.path.join(PSEUDO_LABEL_DIR, f"train_{t}.json"))
        for t in ("mt5_base", "mt5_xlsum")
    )
    if not needed:
        print("Pseudo-labels (BanSum) already exist, skipping generation.\n")
        return True

    print("=" * 80)
    print("STEP 0: GENERATING PSEUDO-LABELS (BanSum Teachers)")
    print("=" * 80)
    cmd = [sys.executable, "generate_pseudo_labels_bansum.py"]
    if quick:
        cmd.append("--quick")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)) or ".")
    if result.returncode != 0:
        print("ERROR: Pseudo-label generation failed!")
        return False
    return True


def find_incomplete_run(config_name):
    """Find the most recent incomplete run (has checkpoints but no ablation_results.json)."""
    if not os.path.exists(RESULTS_DIR):
        return None

    matching = sorted(
        [d for d in os.listdir(RESULTS_DIR) if d.startswith(config_name + "_")],
        reverse=True,
    )

    for run_dir in matching:
        run_path = os.path.join(RESULTS_DIR, run_dir)
        results_file = os.path.join(run_path, "ablation_results.json")

        # Skip if already completed
        if os.path.exists(results_file):
            continue

        # Check for checkpoints
        checkpoints = [d for d in os.listdir(run_path) if d.startswith("checkpoint-")]
        if checkpoints:
            return run_path

    return None


def run_experiment(config_name, quick=False):
    """Run a single ablation experiment as a subprocess (clean GPU memory)."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {config_name}")
    print(f"{'='*80}\n")

    # Check for incomplete run to resume
    incomplete_dir = find_incomplete_run(config_name)

    cmd = [sys.executable, "train_student_ablation_bansum.py", "--config", config_name]
    if quick:
        cmd.append("--quick")
    if incomplete_dir:
        cmd.extend(["--resume_dir", incomplete_dir])
        print(f"Resuming from: {incomplete_dir}\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)) or ".")
    return result.returncode == 0


def find_latest_result(config_name, include_quick=False):
    """Find the most recent results file for a given config."""
    if not os.path.exists(RESULTS_DIR):
        return None

    matching = sorted(
        [d for d in os.listdir(RESULTS_DIR) if d.startswith(config_name + "_")],
        reverse=True,
    )
    if not matching:
        return None

    for run_dir in matching:
        results_file = os.path.join(RESULTS_DIR, run_dir, "ablation_results.json")
        if not os.path.exists(results_file):
            continue
        with open(results_file, "r", encoding="utf-8") as f:
            result = json.load(f)
        if include_quick or not result.get("quick_mode", False):
            return result
    return None


def print_comparison_table(all_results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("ABLATION STUDY RESULTS (BANSUM) — BanglaT5-small Student")
    print("=" * 100)

    header = f"{'Config':<22} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'Gen Len':>8} {'Loss':>8}  Description"
    print(header)
    print("-" * 100)

    baseline_rl = None
    for cfg_name in CONFIGS:
        r = all_results.get(cfg_name)
        if r is None:
            print(f"{cfg_name:<22}  {'—':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>8}  (not run)")
            continue

        tr = r.get("test_results", {})
        r1 = tr.get("test_rouge1", 0)
        r2 = tr.get("test_rouge2", 0)
        rL = tr.get("test_rougeL", 0)
        gl = tr.get("test_gen_len", 0)
        tl = tr.get("test_loss", 0)
        desc = r.get("config", {}).get("description", "")

        if baseline_rl is None:
            baseline_rl = rL

        delta = f" ({rL - baseline_rl:+.4f})" if baseline_rl and cfg_name != "A1_baseline" else ""
        print(f"{cfg_name:<22} {r1:>8.4f} {r2:>8.4f} {rL:>8.4f}{delta:<10} {gl:>5.1f} {tl:>8.4f}  {desc}")

    print("-" * 100)

    # Teacher reference (BanglaT5-BanSum teacher ROUGE-L from test_results.json)
    teacher_rl = 0.2998
    print(f"\nTeacher (BanglaT5-BanSum) ROUGE-L: {teacher_rl:.4f}")
    if baseline_rl:
        print(f"Baseline retention:  {baseline_rl / teacher_rl * 100:.1f}%")
    best_cfg = max(
        [(k, v["test_results"].get("test_rougeL", 0)) for k, v in all_results.items() if v],
        key=lambda x: x[1],
        default=None,
    )
    if best_cfg:
        print(f"Best config:         {best_cfg[0]} (ROUGE-L={best_cfg[1]:.4f}, "
              f"retention={best_cfg[1] / teacher_rl * 100:.1f}%)")

    print()


def save_combined_results(all_results):
    """Save combined ablation results to a single JSON file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "bansum_lte_1000",
        "student_model": "csebuetnlp/banglat5_small",
        "teacher_model": "banglat5_bansum_20260218_213532/final_model (BanglaT5 fine-tuned on BanSum)",
        "pseudo_label_teachers": [
            "mt5base_bansum_20260219_113113/checkpoint-16000 (mT5-base fine-tuned on BanSum)",
            "mt5xlsum_bansum_20260219_062938/checkpoint-14000 (mT5-XLSum fine-tuned on BanSum)",
        ],
        "results": {},
    }

    for cfg_name in CONFIGS:
        r = all_results.get(cfg_name)
        if r:
            output["results"][cfg_name] = {
                "description": r.get("config", {}).get("description", ""),
                "rouge1": r.get("test_results", {}).get("test_rouge1", 0),
                "rouge2": r.get("test_results", {}).get("test_rouge2", 0),
                "rougeL": r.get("test_results", {}).get("test_rougeL", 0),
                "test_loss": r.get("test_results", {}).get("test_loss", 0),
                "train_loss": r.get("train_loss", 0),
            }

    fpath = os.path.join(RESULTS_DIR, "ablation_comparison.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Combined results saved to {fpath}")


def main():
    parser = argparse.ArgumentParser(description="Run full ablation study (BanSum)")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=None,
                        help="Run only specific configs")
    args = parser.parse_args()

    configs_to_run = args.configs or CONFIGS

    print("=" * 80)
    print("ABLATION STUDY (BANSUM) — Enhanced Knowledge Distillation")
    print("=" * 80)
    print(f"Dataset: bansum_lte_1000_tokens.json")
    print(f"Teacher: banglat5_bansum_20260218_213532/final_model")
    print(f"Student: csebuetnlp/banglat5_small")
    print(f"Configs to run: {configs_to_run}")
    print(f"Quick mode: {args.quick}")
    print()

    # Step 0: Pseudo-labels (needed for A3+)
    needs_pseudo = any(c in configs_to_run for c in ("A3_multi_teacher", "A4_adaptive_temp", "A5_full_pipeline"))
    if needs_pseudo:
        if not ensure_pseudo_labels(quick=args.quick):
            print("Cannot proceed without pseudo-labels.")
            sys.exit(1)

    # Run experiments
    successes = []
    failures = []
    skipped = []
    for cfg_name in configs_to_run:
        # Skip if already completed (resume support)
        existing = find_latest_result(cfg_name, include_quick=args.quick)
        if existing and existing.get("test_results", {}).get("test_rougeL", 0) > 0:
            print(f"\n  SKIPPED (already done): {cfg_name} — "
                  f"ROUGE-L={existing['test_results']['test_rougeL']:.4f}")
            skipped.append(cfg_name)
            continue
        ok = run_experiment(cfg_name, quick=args.quick)
        (successes if ok else failures).append(cfg_name)
        print(f"\n  {'SUCCESS' if ok else 'FAILED'}: {cfg_name}")

    # Collect results
    print(f"\n{'='*80}")
    print("COLLECTING RESULTS")
    print(f"{'='*80}")
    all_results = {}
    for cfg_name in CONFIGS:
        all_results[cfg_name] = find_latest_result(cfg_name, include_quick=args.quick)

    # Print comparison
    print_comparison_table(all_results)

    # Save combined
    save_combined_results(all_results)

    # Summary
    print(f"Completed: {len(successes)}/{len(configs_to_run)} | Skipped (already done): {len(skipped)}")
    if failures:
        print(f"Failed: {failures}")


if __name__ == "__main__":
    main()
