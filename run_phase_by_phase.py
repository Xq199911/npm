#!/usr/bin/env python3
"""
MP-KVM Phase-by-Phase Runner with Data Validation

This script allows running individual experimental phases and validating their outputs.
"""
import os
import json
import numpy as np
import argparse
import sys
from pathlib import Path

def validate_phase_1_results():
    """Validate Phase 1 (Baseline Comparison) results."""
    print("Validating Phase 1 Results...")

    results_file = "results/baseline/enhanced_baseline_results.json"
    if not os.path.exists(results_file):
        print("[FAIL] Phase 1 results file not found")
        return False

    try:
        with open(results_file, 'r') as f:
            results = json.load(f)

        if not results:
            print("[FAIL] Phase 1 results are empty")
            return False

        # Check for expected methods (use actual method names from results)
        methods = set(r['method'] for r in results if 'method' in r)
        expected_methods = {'No Compression', 'Random Eviction', 'H2O (Heavy-Hitter)', 'StreamingLLM', 'MP-KVM (Ours)'}

        if not expected_methods.issubset(methods):
            print(f"[FAIL] Missing expected methods. Found: {methods}, Expected: {expected_methods}")
            return False

        # Check for valid metrics
        for result in results:
            if 'recall' not in result or 'compression_ratio' not in result:
                print(f"[FAIL] Missing required metrics in result: {result.get('method', 'Unknown')}")
                return False

            if not (0 <= result['recall'] <= 1):
                print(f"[FAIL] Invalid recall value: {result['recall']} for {result.get('method', 'Unknown')}")
                return False

        print(f"[OK] Phase 1 validation passed: {len(results)} results, {len(methods)} methods")
        return True

    except Exception as e:
        print(f"[ERROR] Error validating Phase 1 results: {e}")
        return False

def validate_phase_2_results():
    """Validate Phase 2 (Manifold Visualization) results."""
    print("Validating Phase 2 Results...")

    # Check for visualization files
    viz_files = [
        "results/figures/manifold_clustering.png",
        "results/figures/topic_transitions.png"
    ]

    missing_files = []
    for viz_file in viz_files:
        if not os.path.exists(viz_file):
            missing_files.append(viz_file)

    if missing_files:
        print(f"[FAIL] Missing visualization files: {missing_files}")
        return False

    # Note: We no longer require synthetic data files since we use real model data
    # Check for manifold data file (may or may not exist with real data)
    data_files = [
        "results/figures/manifold_data.json",
        "results/synthetic/manifold_topic_data.json"
    ]

    data_file_found = False
    for data_file in data_files:
        if os.path.exists(data_file):
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)

                if 'keys' not in data or 'centroids' not in data:
                    print(f"[WARN] {data_file} missing required fields")
                else:
                    keys_shape = data.get('keys_shape', [])
                    centroids_shape = data.get('centroids_shape', [])
                    print(f"[OK] Found manifold data: {keys_shape} keys, {centroids_shape} centroids")
                    data_file_found = True
                    break

            except Exception as e:
                print(f"[WARN] Could not validate {data_file}: {e}")

    if not data_file_found:
        print("[INFO] No manifold data files found (expected with real model data)")

    print("[OK] Phase 2 validation passed: Visualization files generated")
    return True

def validate_phase_3_results():
    """Validate Phase 3 (Core Experiments) results."""
    print("Validating Phase 3 Results...")

    results_file = "results/compression_sweep/compression_sweep_results.json"
    if not os.path.exists(results_file):
        print("[FAIL] Phase 3 results file not found")
        return False

    try:
        with open(results_file, 'r') as f:
            results = json.load(f)

        if not results:
            print("[FAIL] Phase 3 results are empty")
            return False

        # Check for compression sweep data
        compression_ratios = set()
        for result in results:
            if 'compression_ratio' in result:
                compression_ratios.add(result['compression_ratio'])

        if len(compression_ratios) < 3:
            print(f"[FAIL] Insufficient compression ratios tested: {len(compression_ratios)}")
            return False

        print(f"[OK] Phase 3 validation passed: {len(results)} results, {len(compression_ratios)} compression ratios")
        return True

    except Exception as e:
        print(f"[ERROR] Error validating Phase 3 results: {e}")
        return False

def validate_phase_4_results():
    """Validate Phase 4 (Ablation Studies) results."""
    print("Validating Phase 4 Results...")

    results_file = "results/ablation/ablation_results.json"
    if not os.path.exists(results_file):
        print("[FAIL] Phase 4 results file not found")
        return False

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)

        if 'results' not in data:
            print("[FAIL] Phase 4 results missing 'results' field")
            return False

        results = data['results']
        if not results:
            print("[FAIL] Phase 4 results are empty")
            return False

        # Check for ablation parameters
        ablation_configs = set()
        for result in results:
            if 'similarity_threshold' in result and 'max_centroids' in result:
                ablation_configs.add((result['similarity_threshold'], result['max_centroids']))

        if len(ablation_configs) < 4:
            print(f"[FAIL] Insufficient ablation configurations: {len(ablation_configs)}")
            return False

        print(f"[OK] Phase 4 validation passed: {len(results)} ablation results, {len(ablation_configs)} configurations")
        return True

    except Exception as e:
        print(f"[ERROR] Error validating Phase 4 results: {e}")
        return False

def run_phase_1():
    """Run Phase 1: Baseline Comparison."""
    print("Running Phase 1: Baseline Comparison")
    from run_complete_experiment import run_baseline_experiments

    try:
        run_baseline_experiments(use_real_model=True, model_path="model/Llama-3.1-8B-Instruct")
        success = validate_phase_1_results()
        return success
    except Exception as e:
        print(f"[ERROR] Phase 1 failed: {e}")
        return False

def run_phase_2():
    """Run Phase 2: Manifold Visualization."""
    print("Running Phase 2: Manifold Visualization")
    from run_complete_experiment import run_manifold_visualization

    try:
        run_manifold_visualization(use_real_model=True, model_path="model/Llama-3.1-8B-Instruct")
        success = validate_phase_2_results()
        return success
    except Exception as e:
        print(f"[ERROR] Phase 2 failed: {e}")
        return False

def run_phase_3():
    """Run Phase 3: Core Experiments."""
    print("Running Phase 3: Core Experiments")
    from run_complete_experiment import run_core_experiments

    try:
        run_core_experiments(use_real_model=True, model_path="model/Llama-3.1-8B-Instruct")
        success = validate_phase_3_results()
        return success
    except Exception as e:
        print(f"[ERROR] Phase 3 failed: {e}")
        return False

def run_phase_4():
    """Run Phase 4: Ablation Studies."""
    print("Running Phase 4: Ablation Studies")
    from run_complete_experiment import run_ablation_studies

    try:
        run_ablation_studies(use_real_model=True, model_path="model/Llama-3.1-8B-Instruct")
        success = validate_phase_4_results()
        return success
    except Exception as e:
        print(f"[ERROR] Phase 4 failed: {e}")
        return False

def run_phase_5():
    """Run Phase 5: Attention Analysis."""
    print("Running Phase 5: Attention Analysis")
    from run_complete_experiment import run_attention_analysis

    try:
        run_attention_analysis()
        print("[OK] Phase 5 completed (attention analysis)")
        return True
    except Exception as e:
        print(f"[ERROR] Phase 5 failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="MP-KVM Phase-by-Phase Runner with Validation")
    parser.add_argument("phase", type=int, choices=[1, 2, 3, 4, 5],
                       help="Phase to run (1-5)")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate existing results, don't run experiments")

    args = parser.parse_args()

    # Ensure results directories exist
    os.makedirs("results/baseline", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/compression_sweep", exist_ok=True)
    os.makedirs("results/ablation", exist_ok=True)
    os.makedirs("results/attention_analysis", exist_ok=True)

    if args.validate_only:
        print(f"Validating Phase {args.phase} results...")
        validators = {
            1: validate_phase_1_results,
            2: validate_phase_2_results,
            3: validate_phase_3_results,
            4: validate_phase_4_results,
            5: lambda: True  # Phase 5 has no specific validation
        }

        success = validators[args.phase]()
        if success:
            print(f"SUCCESS: Phase {args.phase} validation PASSED")
        else:
            print(f"FAILED: Phase {args.phase} validation FAILED")
        return success

    # Run the specified phase
    runners = {
        1: run_phase_1,
        2: run_phase_2,
        3: run_phase_3,
        4: run_phase_4,
        5: run_phase_5
    }

    print(f"Starting MP-KVM Phase {args.phase}")
    print("=" * 50)

    success = runners[args.phase]()

    print("=" * 50)
    if success:
        print(f"SUCCESS: Phase {args.phase} completed successfully!")
        print("Data validation passed.")
    else:
        print(f"FAILED: Phase {args.phase} failed or validation failed.")
        print("Please check the error messages above.")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
