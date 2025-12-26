#!/usr/bin/env python3
"""
Run key cross-domain experiments with ADBench datasets.

This script runs selected cross-domain anomaly detection experiments
comparing TabPFN vs Padding-based feature extraction.
"""

import os
import sys
import subprocess
import time
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
    
    print(f"\n[RUN] {' + '.join(source_datasets)} -> {target_dataset} ({feature_extractor})")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print(f"[SUCCESS] {' + '.join(source_datasets)} -> {target_dataset} ({feature_extractor})")
            return True
        else:
            print(f"[FAILED] {' + '.join(source_datasets)} -> {target_dataset} ({feature_extractor})")
            if "Error" in result.stderr:
                print(f"Error: {result.stderr[:200]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {' + '.join(source_datasets)} -> {target_dataset} ({feature_extractor})")
        return False
    except Exception as e:
        print(f"[EXCEPTION] {' + '.join(source_datasets)} -> {target_dataset} ({feature_extractor})")
        print(f"Error: {str(e)}")
        return False

def main():
    """Run key cross-domain experiments."""
    print("Starting key cross-domain experiments...")
    print(f"Available datasets: {DATASETS}")
    
    # Define key experiments - select representative combinations
    key_experiments = [
        # Single source experiments - medical datasets
        (['breastw'], 'wdbc'),    # breast cancer -> breast cancer (similar)
        (['breastw'], 'pima'),     # breast cancer -> diabetes (different)
        (['pima'], 'breastw'),     # diabetes -> breast cancer
        (['wdbc'], 'breastw'),     # breast cancer -> breast cancer
        
        # Wine quality experiments
        (['wine'], 'glass'),       # wine -> glass (both material science)
        (['glass'], 'wine'),       # glass -> wine
        
        # Cardiovascular experiments
        (['cardio'], 'pima'),      # cardio -> diabetes (both health)
        (['pima'], 'cardio'),      # diabetes -> cardio
        
        # Liver disease experiments
        (['hepatitis'], 'breastw'), # hepatitis -> breast cancer
        (['hepatitis'], 'cardio'),  # hepatitis -> cardio
        
        # Large dataset experiments
        (['cover'], 'breastw'),     # large -> small dataset
        (['breastw'], 'cover'),     # small -> large dataset
        
        # Multi-source experiments
        (['breastw', 'wdbc'], 'pima'),      # medical -> diabetes
        (['breastw', 'pima'], 'wdbc'),      # medical -> breast cancer
        (['wine', 'glass'], 'breastw'),     # material -> medical
        (['cardio', 'pima'], 'breastw'),    # health -> cancer
    ]
    
    # Feature extractors to test
    feature_extractors = [
        ('tabpfn', None),
        ('padding', 'zero'),
        ('padding', 'mean')
    ]
    
    total_experiments = len(key_experiments) * len(feature_extractors)
    print(f"\nTotal key experiments to run: {total_experiments}")
    print(f"Key experiment combinations: {len(key_experiments)}")
    
    results_summary = []
    successful_experiments = 0
    failed_experiments = 0
    
    start_time = time.time()
    
    for i, (source_datasets, target_dataset) in enumerate(key_experiments):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(key_experiments)}")
        print(f"Sources: {' + '.join(source_datasets)} -> Target: {target_dataset}")
        print(f"{'='*60}")
        
        for j, (feature_extractor, padding_mode) in enumerate(feature_extractors):
            success = run_experiment(
                source_datasets=source_datasets,
                target_dataset=target_dataset,
                feature_extractor=feature_extractor,
                padding_mode=padding_mode,
                epochs=2
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
                'success': success,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Small delay between experiments
            time.sleep(1)
        
        # Progress update every 5 experiments
        if (i + 1) % 5 == 0:
            progress = ((i + 1) / len(key_experiments)) * 100
            elapsed_time = time.time() - start_time
            estimated_total = elapsed_time / (i + 1) * len(key_experiments)
            remaining_time = estimated_total - elapsed_time
            
            print(f"\nProgress: {progress:.1f}% ({i+1}/{len(key_experiments)})")
            print(f"Elapsed: {elapsed_time/60:.1f}m, Estimated remaining: {remaining_time/60:.1f}m")
            print(f"Success: {successful_experiments}, Failed: {failed_experiments}")
    
    # Save experiment summary
    summary_df = pd.DataFrame(results_summary)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    summary_file = f'key_cross_domain_summary_{timestamp}.csv'
    summary_df.to_csv(summary_file, index=False)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"KEY EXPERIMENTS COMPLETED!")
    print(f"{'='*60}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful_experiments}")
    print(f"Failed: {failed_experiments}")
    print(f"Success rate: {successful_experiments/total_experiments*100:.1f}%")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Summary saved to: {summary_file}")
    print(f"Detailed results in: results/results.csv")
    
    # Show statistics
    if successful_experiments > 0:
        print(f"\nStatistics:")
        success_by_extractor = summary_df[summary_df['success']].groupby('feature_extractor').size()
        print(f"   TabPFN successes: {success_by_extractor.get('tabpfn', 0)}")
        print(f"   Padding successes: {success_by_extractor.get('padding', 0)}")
        
        # Show best performing combinations
        if len(summary_df) > 0:
            print(f"\nMost successful source->target pairs:")
            pair_success = summary_df.groupby(['sources', 'target'])['success'].sum().sort_values(ascending=False)
            for (sources, target), success_count in pair_success.head(5).items():
                print(f"   {sources} -> {target}: {success_count}/3 successes")

if __name__ == '__main__':
    main()