"""
=============================================================================
ZT-IPLS Master Experiment Runner
=============================================================================
Runs all three experiments sequentially and produces a consolidated
summary report. Results are saved to results/ and figures/ directories.

Usage
-----
    python run_all_experiments.py           # default seed 42
    python run_all_experiments.py --seed 7  # custom seed

Output
------
    results/experiment1_results.json
    results/experiment2_results.json
    results/experiment3_results.json
    results/consolidated_results.json       ← Section V summary
    figures/                                ← all plots

=============================================================================
"""

import sys
import os
import json
import time
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime

# ── Ensure experiments/ is on the path ────────────────────────────────────────
ROOT = Path(__file__).parent
EXPERIMENTS_DIR = ROOT / 'experiments'
RESULTS_DIR     = ROOT / 'results'
FIGURES_DIR     = ROOT / 'figures'
sys.path.insert(0, str(EXPERIMENTS_DIR))

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def run_experiment(module_name: str, script_path: Path, label: str) -> dict:
    """Import and run an experiment module, capturing its output path."""
    print(f"\n{'='*70}")
    print(f"  Running {label}")
    print(f"{'='*70}")

    # Change to figures dir so plots land there
    original_cwd = os.getcwd()
    os.chdir(str(FIGURES_DIR))

    t0 = time.time()
    try:
        spec   = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
        elapsed = time.time() - t0
        status = 'success'
    except Exception as e:
        elapsed = time.time() - t0
        status = f'error: {e}'
        print(f"\n  ERROR in {label}: {e}")
    finally:
        os.chdir(original_cwd)

    # Move result JSON to results/
    json_name = f'{module_name}_results.json'
    src = FIGURES_DIR / json_name
    dst = RESULTS_DIR / json_name
    if src.exists():
        src.rename(dst)

    return {'label': label, 'status': status, 'elapsed_seconds': round(elapsed, 1)}


def print_consolidated_summary(exp_summaries: list):
    """Print a final summary table across all experiments."""
    print(f"\n{'='*70}")
    print("  ZT-IPLS CONSOLIDATED RESULTS SUMMARY")
    print(f"{'='*70}")

    all_numbers = {}

    for exp_info in exp_summaries:
        label    = exp_info['label']
        json_path = RESULTS_DIR / f"{exp_info['module']}_results.json"
        if not json_path.exists():
            print(f"\n  {label}: result file not found")
            continue

        with open(json_path) as f:
            data = json.load(f)

        nums = data.get('section_v_numbers', {})
        all_numbers[label] = nums

        print(f"\n  {label}")
        print(f"  {'─'*50}")
        for k, v in nums.items():
            if k == 'NOTE':
                continue
            if isinstance(v, float):
                print(f"    {k:<40} {v:.4f}")
            else:
                print(f"    {k:<40} {v}")

    return all_numbers


def main():
    parser = argparse.ArgumentParser(
        description='ZT-IPLS — Run all three experiments'
    )
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--skip', nargs='+', choices=['exp1', 'exp2', 'exp3'],
                        default=[],
                        help='Skip specific experiments')
    args = parser.parse_args()

    print("=" * 70)
    print("  ZT-IPLS: Zero-Trust Architecture for AI-Driven IP Litigation")
    print("  Experimental Evaluation Suite")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Seed: {args.seed}")
    print("=" * 70)

    experiments = [
        {
            'id':     'exp1',
            'module': 'experiment1_prompt_injection',
            'label':  'Experiment 1 — Adversarial Prompt Injection Detection',
            'script': EXPERIMENTS_DIR / 'experiment1_prompt_injection.py',
        },
        {
            'id':     'exp2',
            'module': 'experiment2_rbac',
            'label':  'Experiment 2 — Role-Based Access Control Enforcement',
            'script': EXPERIMENTS_DIR / 'experiment2_rbac.py',
        },
        {
            'id':     'exp3',
            'module': 'experiment3_anomaly_detection',
            'label':  'Experiment 3 — Insider Threat Anomaly Detection',
            'script': EXPERIMENTS_DIR / 'experiment3_anomaly_detection.py',
        },
    ]

    run_log = []
    exp_summaries = []

    for exp in experiments:
        if exp['id'] in args.skip:
            print(f"\n  Skipping {exp['label']} (--skip {exp['id']})")
            continue

        result = run_experiment(exp['module'], exp['script'], exp['label'])
        result['module'] = exp['module']
        run_log.append(result)
        exp_summaries.append({'label': exp['label'], 'module': exp['module']})

    # ── Consolidated summary ──────────────────────────────────────────────────
    all_numbers = print_consolidated_summary(exp_summaries)

    # ── Save consolidated JSON ────────────────────────────────────────────────
    consolidated = {
        'run_timestamp': datetime.now().isoformat(),
        'seed':          args.seed,
        'run_log':       run_log,
        'section_v_numbers': all_numbers,
    }
    out_path = RESULTS_DIR / 'consolidated_results.json'
    with open(out_path, 'w') as f:
        json.dump(consolidated, f, indent=2)

    print(f"\n{'='*70}")
    print("  Run log:")
    for r in run_log:
        status_str = 'OK' if r['status'] == 'success' else f"FAILED ({r['status']})"
        print(f"    {r['label'][:50]:<50} {r['elapsed_seconds']:>6.1f}s  {status_str}")
    print(f"\n  Results saved to: {RESULTS_DIR}")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print(f"  Consolidated:     {out_path}")
    print(f"\n  Send results/consolidated_results.json to write Section V.")
    print("=" * 70)


if __name__ == '__main__':
    main()
