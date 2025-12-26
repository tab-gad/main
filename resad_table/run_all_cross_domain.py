#!/usr/bin/env python3
"""
Run all possible cross-domain experiments with ADBench datasets.

This script runs comprehensive cross-domain anomaly detection experiments
comparing TabPFN vs Padding-based feature extraction across all dataset pairs.
"""

import os
import sys
import subprocess
import time
from itertools import combinations, permutations
import pandas as pd
from pathlib import Path

# Available ADBench datasets
DATASETS = ['breastw', 'pima', 'wdbc', 'wine', 'cardio', 'glass', 'hepatitis', 'cover']

def run_experiment(source_datasets, target_dataset, feature_extractor='tabpfn', padding_mode='zero', epochs=2):
    """Run a single cross-domain experiment."""
    
    cmd = [
        'python', 'resad_tabular.py',
        '--mode', 'cross_domain',
        '--source_datasets'] + source_datasets + [
        '--target_dataset', target_dataset,
        '--feature_extractor', feature_extractor,
        '--epochs', str(epochs)
    ]
    
    if feature_extractor == 'padding':
        cmd.extend(['--padding_mode', padding_mode])
    
    print(f"\n[RUN] Running: {' + '.join(source_datasets)} -> {target_dataset} ({feature_extractor})")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            print(f"[SUCCESS] {' + '.join(source_datasets)} -> {target_dataset} ({feature_extractor})")
            return True
        else:
            print(f"[FAILED] {' + '.join(source_datasets)} -> {target_dataset} ({feature_extractor})")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {' + '.join(source_datasets)} -> {target_dataset} ({feature_extractor})")
        return False
    except Exception as e:
        print(f"[EXCEPTION] {' + '.join(source_datasets)} -> {target_dataset} ({feature_extractor})")
        print(f"Error: {str(e)}")
        return False

def generate_experiment_combinations():
    """Generate all possible cross-domain experiment combinations."""
    experiments = []
    
    # Single source → target experiments
    for source in DATASETS:
        for target in DATASETS:
            if source != target:  # Don't use same dataset as source and target
                experiments.append({
                    'source_datasets': [source],
                    'target_dataset': target,
                    'experiment_type': 'single_source'
                })
    
    # Multiple source → target experiments (2 sources)
    for source_pair in combinations(DATASETS, 2):
        for target in DATASETS:
            if target not in source_pair:
                experiments.append({
                    'source_datasets': list(source_pair),
                    'target_dataset': target,
                    'experiment_type': 'multi_source'
                })
    
    return experiments

def main():
    """Run comprehensive cross-domain experiments."""
    print("Starting comprehensive cross-domain experiments...")
    print(f"Available datasets: {DATASETS}")
    
    # Generate all experiment combinations
    experiments = generate_experiment_combinations()
    
    print(f"\nGenerated {len(experiments)} experiment combinations:")
    print(f"   - Single source experiments: {len([e for e in experiments if e['experiment_type'] == 'single_source'])}")
    print(f"   - Multi source experiments: {len([e for e in experiments if e['experiment_type'] == 'multi_source'])}")
    
    # Run experiments for both feature extractors
    feature_extractors = [
        ('tabpfn', None),
        ('padding', 'zero'),
        ('padding', 'mean')
    ]
    
    total_experiments = len(experiments) * len(feature_extractors)
    print(f"\nTotal experiments to run: {total_experiments}")
    
    results_summary = []
    successful_experiments = 0
    failed_experiments = 0
    
    start_time = time.time()
    
    for i, experiment in enumerate(experiments):
        source_datasets = experiment['source_datasets']
        target_dataset = experiment['target_dataset']
        experiment_type = experiment['experiment_type']
        
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(experiments)}: {experiment_type.upper()}")
        print(f"Sources: {' + '.join(source_datasets)} -> Target: {target_dataset}")
        print(f"{'='*60}")
        
        for feature_extractor, padding_mode in feature_extractors:
            success = run_experiment(
                source_datasets=source_datasets,
                target_dataset=target_dataset,
                feature_extractor=feature_extractor,
                padding_mode=padding_mode,
                epochs=2  # Keep epochs low for speed
            )
            
            if success:
                successful_experiments += 1
            else:
                failed_experiments += 1
            
            results_summary.append({
                'sources': ' + '.join(source_datasets),
                'target': target_dataset,
                'feature_extractor': feature_extractor,
                'padding_mode': padding_mode if feature_extractor == 'padding' else 'N/A',
                'experiment_type': experiment_type,
                'success': success,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Small delay between experiments
            time.sleep(2)
        
        # Progress update
        progress = ((i + 1) / len(experiments)) * 100
        elapsed_time = time.time() - start_time
        estimated_total = elapsed_time / (i + 1) * len(experiments)
        remaining_time = estimated_total - elapsed_time
        
        print(f"\nProgress: {progress:.1f}% ({i+1}/{len(experiments)})")
        print(f"Elapsed: {elapsed_time/60:.1f}m, Estimated remaining: {remaining_time/60:.1f}m")
        print(f"Success: {successful_experiments}, Failed: {failed_experiments}")
    
    # Save experiment summary
    summary_df = pd.DataFrame(results_summary)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    summary_file = f'cross_domain_experiments_summary_{timestamp}.csv'
    summary_df.to_csv(summary_file, index=False)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*60}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful_experiments}")
    print(f"Failed: {failed_experiments}")
    print(f"Success rate: {successful_experiments/total_experiments*100:.1f}%")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Summary saved to: {summary_file}")
    print(f"Detailed results in: results/results.csv")
    
    # Show some statistics
    if successful_experiments > 0:
        print(f"\nQuick Statistics:")
        success_by_extractor = summary_df[summary_df['success']].groupby('feature_extractor').size()
        print(f"   TabPFN successes: {success_by_extractor.get('tabpfn', 0)}")
        print(f"   Padding successes: {success_by_extractor.get('padding', 0)}")
        
        success_by_type = summary_df[summary_df['success']].groupby('experiment_type').size()
        print(f"   Single source successes: {success_by_type.get('single_source', 0)}")
        print(f"   Multi source successes: {success_by_type.get('multi_source', 0)}")

if __name__ == '__main__':
    main()