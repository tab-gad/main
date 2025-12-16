#!/usr/bin/env python3
"""
Complete ResAD Tabular Pipeline Execution Script

This script runs the complete ResAD for tabular data pipeline including:
1. Dataset preparation and loading
2. TabPFN feature extraction
3. Feature constraint learning  
4. Normal distribution estimation
5. Anomaly detection evaluation
6. Cross-domain and ablation experiments

Usage:
    python run_all.py --experiment single_domain --dataset breastw
    python run_all.py --experiment cross_domain --source_datasets breastw pima --target_dataset wdbc
    python run_all.py --experiment ablation
"""

import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from resad_tabular import train_resad, parse_args
from datasets.tabular_loader import ADBenchDataLoader


class ResADPipelineRunner:
    """Complete pipeline runner for ResAD tabular experiments."""
    
    def __init__(self, config):
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Pipeline run directory: {self.run_dir}")
        
        # Available datasets for experiments
        self.available_datasets = [
            'breastw', 'pima', 'wdbc', 'wine', 'cardio', 
            'glass', 'hepatitis', 'cover'
        ]
        
        # Results storage
        self.all_results = {}
    
    def run_single_experiment(self, dataset, mode='single_domain', **kwargs):
        """Run a single ResAD experiment."""
        print(f"\n{'='*60}")
        print(f"Running ResAD experiment: {dataset} ({mode})")
        print(f"{'='*60}")
        
        # Prepare arguments
        args = argparse.Namespace()
        
        # Copy default config
        for key, value in vars(self.config).items():
            setattr(args, key, value)
        
        # Set experiment-specific arguments
        args.dataset = dataset
        args.mode = mode
        args.save_dir = str(self.run_dir)
        
        # Apply any additional kwargs
        for key, value in kwargs.items():
            setattr(args, key, value)
        
        start_time = time.time()
        
        try:
            # Run training
            model, results = train_resad(args)
            
            execution_time = time.time() - start_time
            results['execution_time'] = execution_time
            
            # Store results
            result_key = f"{dataset}_{mode}"
            if 'source_datasets' in kwargs:
                source_str = '_'.join(kwargs['source_datasets'])
                result_key = f"{source_str}_to_{dataset}"
            
            self.all_results[result_key] = results
            
            print(f"Experiment completed successfully in {execution_time:.2f} seconds")
            print(f"AUC: {results['auc']:.4f}, AP: {results['ap']:.4f}")
            
            return results
            
        except Exception as e:
            print(f"Experiment failed: {str(e)}")
            error_result = {
                'error': str(e),
                'execution_time': time.time() - start_time,
                'config': vars(args)
            }
            self.all_results[f"{dataset}_{mode}_ERROR"] = error_result
            return None
    
    def run_single_domain_experiments(self):
        """Run single-domain experiments on all datasets."""
        print(f"\n{'='*80}")
        print("RUNNING SINGLE-DOMAIN EXPERIMENTS")
        print(f"{'='*80}")
        
        single_domain_results = {}
        
        for dataset in self.available_datasets:
            if hasattr(self.config, 'datasets_filter') and dataset not in self.config.datasets_filter:
                continue
                
            result = self.run_single_experiment(dataset, mode='single_domain')
            if result:
                single_domain_results[dataset] = result
        
        # Save aggregated results
        self._save_single_domain_summary(single_domain_results)
        
        return single_domain_results
    
    def run_cross_domain_experiments(self):
        """Run cross-domain experiments."""
        print(f"\n{'='*80}")
        print("RUNNING CROSS-DOMAIN EXPERIMENTS")
        print(f"{'='*80}")
        
        # Define cross-domain pairs
        cross_domain_pairs = [
            (['breastw', 'pima'], 'wdbc'),
            (['breastw', 'wdbc'], 'pima'),
            (['pima', 'wdbc'], 'breastw'),
            (['cardio', 'glass'], 'wine'),
            (['wine', 'glass'], 'cardio'),
        ]
        
        cross_domain_results = {}
        
        for source_datasets, target_dataset in cross_domain_pairs:
            if hasattr(self.config, 'datasets_filter'):
                # Skip if any dataset not in filter
                all_datasets = source_datasets + [target_dataset]
                if not all(d in self.config.datasets_filter for d in all_datasets):
                    continue
            
            result = self.run_single_experiment(
                target_dataset,
                mode='cross_domain',
                source_datasets=source_datasets,
                target_dataset=target_dataset
            )
            
            if result:
                key = f"{'_'.join(source_datasets)}_to_{target_dataset}"
                cross_domain_results[key] = result
        
        # Save aggregated results
        self._save_cross_domain_summary(cross_domain_results)
        
        return cross_domain_results
    
    def run_ablation_studies(self):
        """Run ablation studies on different components."""
        print(f"\n{'='*80}")
        print("RUNNING ABLATION STUDIES")
        print(f"{'='*80}")
        
        base_dataset = self.config.ablation_dataset if hasattr(self.config, 'ablation_dataset') else 'breastw'
        
        ablation_configs = {
            'baseline': {},
            'no_multiscale': {'use_multiscale': False},
            'basic_constraintor': {'constraintor_type': 'basic'},
            'adaptive_constraintor': {'constraintor_type': 'adaptive'},
            'flow_estimator': {'estimator_type': 'flow'},
            'ensemble_estimator': {'estimator_type': 'ensemble'},
            'reduced_epochs': {'epochs': 20},
            'higher_lr': {'lr': 5e-3},
        }
        
        ablation_results = {}
        
        for config_name, config_changes in ablation_configs.items():
            print(f"\nRunning ablation: {config_name}")
            
            result = self.run_single_experiment(
                base_dataset,
                mode='single_domain',
                **config_changes
            )
            
            if result:
                ablation_results[config_name] = result
        
        # Save ablation results
        self._save_ablation_summary(ablation_results)
        
        return ablation_results
    
    def run_parameter_sensitivity(self):
        """Run parameter sensitivity analysis."""
        print(f"\n{'='*80}")
        print("RUNNING PARAMETER SENSITIVITY ANALYSIS")
        print(f"{'='*80}")
        
        base_dataset = self.config.sensitivity_dataset if hasattr(self.config, 'sensitivity_dataset') else 'breastw'
        
        # Parameter ranges to test
        param_ranges = {
            'lr': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            'batch_size': [16, 32, 64, 128],
            'epochs': [20, 30, 50, 70, 100],
            'reg_weight': [1e-5, 1e-4, 1e-3, 1e-2],
            'n_fold': [0, 3, 5, 7],
        }
        
        sensitivity_results = {}
        
        for param_name, values in param_ranges.items():
            param_results = {}
            
            for value in values:
                config_name = f"{param_name}_{value}"
                print(f"\nTesting {param_name} = {value}")
                
                result = self.run_single_experiment(
                    base_dataset,
                    mode='single_domain',
                    **{param_name: value}
                )
                
                if result:
                    param_results[str(value)] = result
            
            sensitivity_results[param_name] = param_results
        
        # Save sensitivity results
        self._save_sensitivity_summary(sensitivity_results)
        
        return sensitivity_results
    
    def _save_single_domain_summary(self, results):
        """Save single-domain results summary."""
        summary_data = []
        
        for dataset, result in results.items():
            summary_data.append({
                'dataset': dataset,
                'auc': result['auc'],
                'ap': result['ap'],
                'execution_time': result['execution_time']
            })
        
        df = pd.DataFrame(summary_data)
        
        # Save CSV
        df.to_csv(self.run_dir / 'single_domain_results.csv', index=False)
        
        # Create visualization
        self._plot_single_domain_results(df)
        
        # Print summary
        print("\n" + "="*60)
        print("SINGLE-DOMAIN RESULTS SUMMARY")
        print("="*60)
        print(df.to_string(index=False))
        print(f"\nMean AUC: {df['auc'].mean():.4f} ± {df['auc'].std():.4f}")
        print(f"Mean AP: {df['ap'].mean():.4f} ± {df['ap'].std():.4f}")
    
    def _save_cross_domain_summary(self, results):
        """Save cross-domain results summary."""
        summary_data = []
        
        for experiment, result in results.items():
            parts = experiment.split('_to_')
            source = parts[0]
            target = parts[1]
            
            summary_data.append({
                'source_datasets': source,
                'target_dataset': target,
                'auc': result['auc'],
                'ap': result['ap'],
                'execution_time': result['execution_time']
            })
        
        df = pd.DataFrame(summary_data)
        
        # Save CSV
        df.to_csv(self.run_dir / 'cross_domain_results.csv', index=False)
        
        # Create visualization
        self._plot_cross_domain_results(df)
        
        # Print summary
        print("\n" + "="*60)
        print("CROSS-DOMAIN RESULTS SUMMARY")
        print("="*60)
        print(df.to_string(index=False))
        print(f"\nMean AUC: {df['auc'].mean():.4f} ± {df['auc'].std():.4f}")
        print(f"Mean AP: {df['ap'].mean():.4f} ± {df['ap'].std():.4f}")
    
    def _save_ablation_summary(self, results):
        """Save ablation study results."""
        summary_data = []
        
        for config_name, result in results.items():
            summary_data.append({
                'configuration': config_name,
                'auc': result['auc'],
                'ap': result['ap'],
                'execution_time': result['execution_time']
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('auc', ascending=False)
        
        # Save CSV
        df.to_csv(self.run_dir / 'ablation_results.csv', index=False)
        
        # Create visualization
        self._plot_ablation_results(df)
        
        # Print summary
        print("\n" + "="*60)
        print("ABLATION STUDY RESULTS")
        print("="*60)
        print(df.to_string(index=False))
    
    def _save_sensitivity_summary(self, results):
        """Save parameter sensitivity results."""
        # Save detailed results
        with open(self.run_dir / 'sensitivity_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary plots
        self._plot_sensitivity_results(results)
        
        print("\n" + "="*60)
        print("PARAMETER SENSITIVITY ANALYSIS COMPLETED")
        print("="*60)
        print(f"Results saved to: {self.run_dir}")
    
    def _plot_single_domain_results(self, df):
        """Create plots for single-domain results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # AUC plot
        bars1 = ax1.bar(df['dataset'], df['auc'], color='skyblue', alpha=0.7)
        ax1.set_title('AUC by Dataset')
        ax1.set_ylabel('AUC')
        ax1.set_xlabel('Dataset')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, df['auc']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # AP plot
        bars2 = ax2.bar(df['dataset'], df['ap'], color='lightcoral', alpha=0.7)
        ax2.set_title('Average Precision by Dataset')
        ax2.set_ylabel('AP')
        ax2.set_xlabel('Dataset')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars2, df['ap']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'single_domain_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cross_domain_results(self, df):
        """Create plots for cross-domain results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Create experiment labels
        labels = [f"{row['source_datasets']} → {row['target_dataset']}" 
                 for _, row in df.iterrows()]
        
        # AUC plot
        bars1 = ax1.bar(range(len(labels)), df['auc'], color='lightgreen', alpha=0.7)
        ax1.set_title('Cross-Domain AUC')
        ax1.set_ylabel('AUC')
        ax1.set_xlabel('Source → Target')
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars1, df['auc']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # AP plot
        bars2 = ax2.bar(range(len(labels)), df['ap'], color='lightsalmon', alpha=0.7)
        ax2.set_title('Cross-Domain Average Precision')
        ax2.set_ylabel('AP')
        ax2.set_xlabel('Source → Target')
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars2, df['ap']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'cross_domain_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ablation_results(self, df):
        """Create plots for ablation results."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create horizontal bar plot
        bars = ax.barh(df['configuration'], df['auc'], color='gold', alpha=0.7)
        ax.set_title('Ablation Study Results (AUC)')
        ax.set_xlabel('AUC')
        ax.set_ylabel('Configuration')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, df['auc']):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', va='center', ha='left')
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'ablation_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sensitivity_results(self, results):
        """Create plots for parameter sensitivity."""
        n_params = len(results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (param_name, param_results) in enumerate(results.items()):
            if i >= len(axes):
                break
            
            values = []
            aucs = []
            
            for value_str, result in param_results.items():
                try:
                    value = float(value_str)
                    values.append(value)
                    aucs.append(result['auc'])
                except:
                    continue
            
            if values and aucs:
                # Sort by parameter value
                sorted_data = sorted(zip(values, aucs))
                values, aucs = zip(*sorted_data)
                
                axes[i].plot(values, aucs, 'o-', color='purple', alpha=0.7)
                axes[i].set_title(f'Parameter Sensitivity: {param_name}')
                axes[i].set_xlabel(param_name)
                axes[i].set_ylabel('AUC')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'sensitivity_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_full_pipeline(self):
        """Run the complete ResAD pipeline with all experiments."""
        print(f"{'='*100}")
        print("STARTING COMPLETE RESAD TABULAR PIPELINE")
        print(f"{'='*100}")
        print(f"Results will be saved to: {self.run_dir}")
        print(f"Available datasets: {self.available_datasets}")
        
        start_time = time.time()
        
        # Run experiments based on configuration
        if self.config.experiment == 'single_domain' or self.config.experiment == 'all':
            self.run_single_domain_experiments()
        
        if self.config.experiment == 'cross_domain' or self.config.experiment == 'all':
            self.run_cross_domain_experiments()
        
        if self.config.experiment == 'ablation' or self.config.experiment == 'all':
            self.run_ablation_studies()
        
        if self.config.experiment == 'sensitivity' or self.config.experiment == 'all':
            self.run_parameter_sensitivity()
        
        # Save complete results
        with open(self.run_dir / 'complete_results.json', 'w') as f:
            json.dump(self.all_results, f, indent=2)
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*100}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*100}")
        print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Results saved to: {self.run_dir}")
        print(f"Number of experiments: {len(self.all_results)}")


def main():
    """Main function to run the complete pipeline."""
    parser = argparse.ArgumentParser(description='ResAD Tabular Complete Pipeline')
    
    # Add all arguments from resad_tabular
    parser.add_argument('--data_root', type=str, default='../dataset/Classical',
                        help='Root directory for datasets')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save all results')
    
    # Experiment configuration
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['single_domain', 'cross_domain', 'ablation', 'sensitivity', 'all'],
                        help='Type of experiment to run')
    parser.add_argument('--datasets_filter', nargs='+', default=None,
                        help='Filter to run only specific datasets')
    parser.add_argument('--ablation_dataset', type=str, default='breastw',
                        help='Dataset to use for ablation studies')
    parser.add_argument('--sensitivity_dataset', type=str, default='breastw',
                        help='Dataset to use for sensitivity analysis')
    
    # Model configuration (defaults)
    parser.add_argument('--use_multiscale', action='store_true', default=True,
                        help='Use multi-scale feature extractor')
    parser.add_argument('--n_scales', type=int, default=3,
                        help='Number of scales for multi-scale extractor')
    parser.add_argument('--n_estimators', type=int, default=1,
                        help='Number of TabPFN estimators')
    parser.add_argument('--n_fold', type=int, default=5,
                        help='Number of folds for TabPFN cross-validation')
    parser.add_argument('--use_scaler', action='store_true', default=True,
                        help='Use feature scaling')
    parser.add_argument('--final_embedding_dim', type=int, default=128,
                        help='Final embedding dimension')
    parser.add_argument('--constraintor_type', type=str, default='residual',
                        choices=['basic', 'residual', 'adaptive'],
                        help='Type of feature constraintor')
    parser.add_argument('--num_residual_blocks', type=int, default=3,
                        help='Number of residual blocks')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--estimator_type', type=str, default='normal',
                        choices=['normal', 'flow', 'ensemble'],
                        help='Type of distribution estimator')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--reg_weight', type=float, default=1e-4,
                        help='Regularization weight')
    parser.add_argument('--test_size', type=float, default=0.3,
                        help='Test set proportion')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = ResADPipelineRunner(args)
    pipeline.run_full_pipeline()


if __name__ == '__main__':
    main()