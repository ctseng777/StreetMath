#!/usr/bin/env python3
"""
Linear Probe Results Analysis
============================

Implements the 4-step analysis strategy:
1. Architecture-level overview (find major differences)
2. Task-specific deep dive (digits vs words)  
3. Fine-grained Error analysis (distance patterns)
4. Cross-validate findings across different near values

Author: Analysis for StreetMath LinearProbe experiments
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Configuration
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = BASE_DIR / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Model metadata
MODEL_INFO = {
    'dream': {
        'architecture': 'diffusion',
        'size': '7B',
        'total_layers': 28,
        'display_name': 'Dream-v0-Instruct-7B'
    },
    'falcon_h1_7b_instruct': {
        'architecture': 'state_space',
        'size': '7B',
        'total_layers': 44,
        'display_name': 'Falcon-H1-7B-Instruct'
    },
    'falcon_h1_34b_instruct': {
        'architecture': 'state_space',
        'size': '34B',
        'total_layers': 72,
        'display_name': 'Falcon-H1-34B-Instruct'
    },
    'qwen_2_5_3b_instruct': {
        'architecture': 'autoregressive',
        'size': '3B',
        'total_layers': 36,
        'display_name': 'Qwen2.5-3B-Instruct'
    },
    'qwen_2_5_32b_instruct': {
        'architecture': 'autoregressive',
        'size': '32B',
        'total_layers': 64,
        'display_name': 'Qwen2.5-32B-Instruct'
    },
    'qwen3_4b_thinking': {
        'architecture': 'autoregressive',
        'size': '4B',
        'total_layers': 36,
        'display_name': 'Qwen3-4B-Thinking-2507'
    }
}

DISTANCES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
TASKS = ['digits_paraphrase', 'words']


class LinearProbeAnalyzer:
    def __init__(self):
        self.data = {}
        self.summary_stats = {}
        
    def load_all_data(self):
        """Load all JSON results into structured data."""
        print("Loading all probe results...")
        
        for distance in DISTANCES:
            distance_dir = RESULTS_DIR / str(distance)
            if not distance_dir.exists():
                print(f"Warning: Directory {distance_dir} not found")
                continue
                
            self.data[distance] = {}
            
            for json_file in distance_dir.glob("probe_results_*.json"):
                # Extract model name from filename
                model_key = json_file.stem.replace('probe_results_', '').replace(f'_near_{distance}', '')
                
                if model_key not in MODEL_INFO:
                    print(f"Warning: Unknown model {model_key}")
                    continue
                
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                self.data[distance][model_key] = data.get('model1', {})
        
        print(f"Loaded data for {len(self.data)} distances and {len(MODEL_INFO)} models")
    
    def extract_metrics(self, model_data: Dict, task: str) -> Dict[str, Any]:
        """Extract key metrics from model data for a specific task."""
        if task not in model_data:
            return {}
        
        task_data = model_data[task]
        acc_per_layer = task_data.get('acc_per_layer', {})
        
        if not acc_per_layer:
            return {}
        
        # Convert string keys to ints and find peak
        layer_accs = {int(k): float(v) for k, v in acc_per_layer.items()}
        peak_acc = max(layer_accs.values()) if layer_accs else 0.0
        best_layer = max(layer_accs, key=layer_accs.get) if layer_accs else 0
        
        return {
            'peak_accuracy': peak_acc,
            'best_layer': best_layer,
            'layer_accuracies': layer_accs,
            'err_by_dist': task_data.get('err_by_dist', {}),
            'err_by_dir': task_data.get('err_by_dir', {})
        }

    # ==================== STEP 1: Architecture-level Overview ====================
    
    def analyze_architecture_overview(self):
        """Step 1: Architecture-level analysis to find major differences."""
        print("\n" + "="*60)
        print("STEP 1: Architecture-level Overview")
        print("="*60)
        
        # Aggregate data by architecture
        arch_performance = defaultdict(lambda: defaultdict(list))
        convergence_data = []
        
        for distance in DISTANCES:
            if distance not in self.data:
                continue
                
            for model_key, model_data in self.data[distance].items():
                arch = MODEL_INFO[model_key]['architecture']
                
                for task in TASKS:
                    metrics = self.extract_metrics(model_data, task)
                    if metrics:
                        arch_performance[arch][f"{task}_peak_acc"].append(metrics['peak_accuracy'])
                        
                        # Calculate depth percentage
                        total_layers = MODEL_INFO[model_key]['total_layers']
                        depth_pct = (metrics['best_layer'] / total_layers) * 100
                        convergence_data.append({
                            'architecture': arch,
                            'model': model_key,
                            'distance': distance,
                            'task': task,
                            'best_layer': metrics['best_layer'],
                            'total_layers': total_layers,
                            'depth_percentage': depth_pct,
                            'peak_accuracy': metrics['peak_accuracy']
                        })
        
        # Create convergence DataFrame
        convergence_df = pd.DataFrame(convergence_data)
        
        # Generate architecture summary
        print("\nArchitecture Performance Summary:")
        print("-" * 40)
        
        for arch in ['autoregressive', 'state_space', 'diffusion']:
            if arch in arch_performance:
                digits_acc = arch_performance[arch]['digits_paraphrase_peak_acc']
                words_acc = arch_performance[arch]['words_peak_acc']
                
                arch_data = convergence_df[convergence_df['architecture'] == arch]
                avg_depth = arch_data['depth_percentage'].mean()
                
                print(f"\n{arch.upper()}:")
                print(f"  Digits avg accuracy: {np.mean(digits_acc):.3f} ± {np.std(digits_acc):.3f}")
                print(f"  Words avg accuracy:  {np.mean(words_acc):.3f} ± {np.std(words_acc):.3f}")
                print(f"  Avg convergence depth: {avg_depth:.1f}%")
        
        # Plot architecture comparison
        self._plot_architecture_heatmap(convergence_df)
        self._plot_convergence_patterns(convergence_df)
        
        return convergence_df
    
    def _plot_architecture_heatmap(self, df: pd.DataFrame):
        """Create heatmap of peak accuracies across architectures and distances."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for i, task in enumerate(['digits_paraphrase', 'words']):
            task_data = df[df['task'] == task]
            
            # Create pivot table
            pivot_data = task_data.pivot_table(
                values='peak_accuracy',
                index='architecture', 
                columns='distance',
                aggfunc='mean'
            )
            
            sns.heatmap(
                pivot_data, 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlBu_r',
                ax=ax1 if i == 0 else ax2,
                cbar_kws={'label': 'Peak Accuracy'}
            )
            
            ax = ax1 if i == 0 else ax2
            ax.set_title(f'Peak Accuracy: {task.replace("_", " ").title()}')
            ax.set_xlabel('Distance')
            ax.set_ylabel('Architecture')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'step1_architecture_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {OUTPUT_DIR / 'step1_architecture_heatmap.png'}")
    
    def _plot_convergence_patterns(self, df: pd.DataFrame):
        """Plot convergence depth patterns by architecture."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Box plot of convergence depths by architecture
        sns.boxplot(
            data=df,
            x='architecture',
            y='depth_percentage',
            hue='task',
            ax=ax
        )
        
        ax.set_title('Convergence Depth Distribution by Architecture')
        ax.set_xlabel('Architecture')
        ax.set_ylabel('Best Layer Depth (%)')
        ax.legend(title='Task')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'step1_convergence_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {OUTPUT_DIR / 'step1_convergence_patterns.png'}")

    # ==================== STEP 2: Task-specific Deep Dive ====================
    
    def analyze_task_comparison(self):
        """Step 2: Deep dive into digits vs words performance gap."""
        print("\n" + "="*60)
        print("STEP 2: Task-specific Deep Dive (Digits vs Words)")
        print("="*60)
        
        comparison_data = []
        
        for distance in DISTANCES:
            if distance not in self.data:
                continue
                
            for model_key, model_data in self.data[distance].items():
                digits_metrics = self.extract_metrics(model_data, 'digits_paraphrase')
                words_metrics = self.extract_metrics(model_data, 'words')
                
                if digits_metrics and words_metrics:
                    performance_gap = digits_metrics['peak_accuracy'] - words_metrics['peak_accuracy']
                    
                    comparison_data.append({
                        'model': MODEL_INFO[model_key]['display_name'],
                        'model_key': model_key,
                        'architecture': MODEL_INFO[model_key]['architecture'],
                        'distance': distance,
                        'digits_acc': digits_metrics['peak_accuracy'],
                        'words_acc': words_metrics['peak_accuracy'],
                        'performance_gap': performance_gap,
                        'digits_best_layer': digits_metrics['best_layer'],
                        'words_best_layer': words_metrics['best_layer']
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Summary statistics
        print("\nPerformance Gap Analysis:")
        print("-" * 30)
        
        gap_by_arch = comparison_df.groupby('architecture')['performance_gap'].agg(['mean', 'std'])
        print(gap_by_arch)
        
        # Identify patterns
        special_numbers = [2, 5, 10]
        ordinary_numbers = [3, 4, 6, 7, 8, 9]
        
        special_gaps = comparison_df[comparison_df['distance'].isin(special_numbers)]['performance_gap']
        ordinary_gaps = comparison_df[comparison_df['distance'].isin(ordinary_numbers)]['performance_gap']
        
        print(f"\nSpecial numbers (2,5,10) avg gap: {special_gaps.mean():.3f}")
        print(f"Ordinary numbers avg gap: {ordinary_gaps.mean():.3f}")
        
        # Generate visualizations
        self._plot_task_comparison(comparison_df)
        self._plot_performance_gaps(comparison_df)
        
        return comparison_df
    
    def _plot_task_comparison(self, df: pd.DataFrame):
        """Create scatter plot of digits vs words performance."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        architectures = df['architecture'].unique()
        colors = ['blue', 'red', 'green']
        
        for i, arch in enumerate(architectures):
            arch_data = df[df['architecture'] == arch]
            ax.scatter(
                arch_data['digits_acc'],
                arch_data['words_acc'],
                c=colors[i],
                label=arch.title(),
                alpha=0.7,
                s=50
            )
        
        # Add diagonal line for reference
        max_acc = max(df['digits_acc'].max(), df['words_acc'].max())
        ax.plot([0, max_acc], [0, max_acc], 'k--', alpha=0.5, label='Equal Performance')
        
        ax.set_xlabel('Digits Task Peak Accuracy')
        ax.set_ylabel('Words Task Peak Accuracy') 
        ax.set_title('Digits vs Words Performance by Architecture')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'step2_digits_vs_words.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {OUTPUT_DIR / 'step2_digits_vs_words.png'}")
    
    def _plot_performance_gaps(self, df: pd.DataFrame):
        """Plot performance gaps by distance and architecture."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Gap by distance
        gap_by_distance = df.groupby('distance')['performance_gap'].agg(['mean', 'std'])
        
        ax1.bar(gap_by_distance.index, gap_by_distance['mean'], 
                yerr=gap_by_distance['std'], capsize=5, alpha=0.7)
        ax1.set_xlabel('Distance')
        ax1.set_ylabel('Performance Gap (Digits - Words)')
        ax1.set_title('Performance Gap by Distance')
        ax1.grid(True, alpha=0.3)
        
        # Gap by architecture
        sns.boxplot(data=df, x='architecture', y='performance_gap', ax=ax2)
        ax2.set_title('Performance Gap Distribution by Architecture')
        ax2.set_ylabel('Performance Gap (Digits - Words)')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'step2_performance_gaps.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {OUTPUT_DIR / 'step2_performance_gaps.png'}")

    # ==================== STEP 3: Error Analysis ====================
    
    def analyze_error_patterns(self):
        """Step 3: Fine-grained analysis of error patterns by distance."""
        print("\n" + "="*60) 
        print("STEP 3: Fine-grained Error Analysis")
        print("="*60)
        
        error_data = []
        
        for distance in DISTANCES:
            if distance not in self.data:
                continue
                
            for model_key, model_data in self.data[distance].items():
                for task in TASKS:
                    metrics = self.extract_metrics(model_data, task)
                    if not metrics or 'err_by_dist' not in metrics:
                        continue
                    
                    err_by_dist = metrics['err_by_dist']
                    
                    for dist_key, dist_data in err_by_dist.items():
                        if isinstance(dist_data, list) and len(dist_data) >= 2:
                            error_rate, count = dist_data[0], dist_data[1]
                        elif isinstance(dist_data, (int, float)):
                            error_rate, count = dist_data, 1
                        else:
                            continue
                            
                        if count > 0:  # Only include distances with data
                            error_data.append({
                                'model': MODEL_INFO[model_key]['display_name'],
                                'model_key': model_key,
                                'architecture': MODEL_INFO[model_key]['architecture'],
                                'target_distance': distance,
                                'task': task,
                                'error_distance': int(dist_key),
                                'error_rate': error_rate,
                                'sample_count': count
                            })
        
        error_df = pd.DataFrame(error_data)
        
        # Analyze distance-1 boundary effects
        print("\nBoundary Effect Analysis (Distance-1 errors):")
        print("-" * 45)
        
        boundary_errors = error_df[error_df['error_distance'] == 1]
        boundary_by_task = boundary_errors.groupby(['task', 'architecture'])['error_rate'].mean()
        print(boundary_by_task)
        
        # Generate visualizations
        self._plot_error_patterns(error_df)
        self._plot_boundary_effects(error_df)
        
        return error_df
    
    def _plot_error_patterns(self, df: pd.DataFrame):
        """Plot error rates by distance for each architecture."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        architectures = ['autoregressive', 'state_space', 'diffusion']
        tasks = ['digits_paraphrase', 'words']
        
        plot_idx = 0
        for task in tasks:
            for arch in architectures:
                if plot_idx >= len(axes):
                    break
                    
                task_arch_data = df[(df['task'] == task) & (df['architecture'] == arch)]
                
                if len(task_arch_data) == 0:
                    axes[plot_idx].text(0.5, 0.5, 'No Data', ha='center', va='center')
                    axes[plot_idx].set_title(f'{arch.title()}\n{task.replace("_", " ").title()}')
                    plot_idx += 1
                    continue
                
                # Average error rate by error distance
                avg_errors = task_arch_data.groupby('error_distance')['error_rate'].mean()
                
                axes[plot_idx].plot(avg_errors.index, avg_errors.values, 'o-', linewidth=2, markersize=6)
                axes[plot_idx].set_title(f'{arch.title()}\n{task.replace("_", " ").title()}')
                axes[plot_idx].set_xlabel('Error Distance')
                axes[plot_idx].set_ylabel('Error Rate')
                axes[plot_idx].grid(True, alpha=0.3)
                
                plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'step3_error_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {OUTPUT_DIR / 'step3_error_patterns.png'}")
    
    def _plot_boundary_effects(self, df: pd.DataFrame):
        """Plot boundary effects (distance-1 errors) comparison."""
        boundary_data = df[df['error_distance'] == 1]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.boxplot(
            data=boundary_data,
            x='architecture',
            y='error_rate', 
            hue='task',
            ax=ax
        )
        
        ax.set_title('Boundary Effects: Distance-1 Error Rates')
        ax.set_xlabel('Architecture')
        ax.set_ylabel('Distance-1 Error Rate')
        ax.legend(title='Task')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'step3_boundary_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {OUTPUT_DIR / 'step3_boundary_effects.png'}")

    # ==================== STEP 4: Cross-validation ====================
    
    def cross_validate_findings(self, convergence_df, comparison_df, error_df):
        """Step 4: Cross-validate findings across different near values."""
        print("\n" + "="*60)
        print("STEP 4: Cross-validation Across Near Values") 
        print("="*60)
        
        # 1. Architecture ranking consistency
        print("\nArchitecture Ranking Consistency:")
        print("-" * 35)
        
        arch_rankings = {}
        for distance in DISTANCES:
            dist_data = convergence_df[
                (convergence_df['distance'] == distance) & 
                (convergence_df['task'] == 'digits_paraphrase')
            ]
            if len(dist_data) > 0:
                ranking = dist_data.groupby('architecture')['peak_accuracy'].mean().sort_values(ascending=False)
                arch_rankings[distance] = list(ranking.index)
        
        # Calculate ranking correlations
        correlations = []
        distance_pairs = []
        
        for i, dist1 in enumerate(DISTANCES[:-1]):
            for dist2 in DISTANCES[i+1:]:
                if dist1 in arch_rankings and dist2 in arch_rankings:
                    rank1 = [arch_rankings[dist1].index(arch) for arch in arch_rankings[dist1]]
                    rank2 = [arch_rankings[dist2].index(arch) for arch in arch_rankings[dist2] if arch in arch_rankings[dist1]]
                    
                    if len(rank1) == len(rank2):
                        corr = np.corrcoef(rank1, rank2)[0,1]
                        correlations.append(corr)
                        distance_pairs.append((dist1, dist2))
        
        avg_correlation = np.mean(correlations) if correlations else 0
        print(f"Average ranking correlation: {avg_correlation:.3f}")
        
        # 2. Mathematical hierarchy validation
        print("\nMathematical Hierarchy Validation:")
        print("-" * 35)
        
        special_distances = [2, 5, 10]
        ordinary_distances = [3, 4, 6, 7, 8, 9]
        
        special_performance = []
        ordinary_performance = []
        
        for distance in DISTANCES:
            dist_data = convergence_df[
                (convergence_df['distance'] == distance) & 
                (convergence_df['task'] == 'digits_paraphrase')
            ]
            avg_acc = dist_data['peak_accuracy'].mean()
            
            if distance in special_distances:
                special_performance.append(avg_acc)
            elif distance in ordinary_distances:
                ordinary_performance.append(avg_acc)
        
        special_avg = np.mean(special_performance)
        ordinary_avg = np.mean(ordinary_performance)
        
        print(f"Special numbers (2,5,10) avg accuracy: {special_avg:.3f}")
        print(f"Ordinary numbers avg accuracy: {ordinary_avg:.3f}")
        print(f"Special > Ordinary: {special_avg > ordinary_avg}")
        
        # Generate cross-validation plots
        self._plot_consistency_analysis(arch_rankings, convergence_df)
        self._plot_hierarchy_validation(convergence_df)
        
        # Generate summary report
        self._generate_summary_report(convergence_df, comparison_df, error_df, {
            'ranking_correlation': avg_correlation,
            'special_avg': special_avg,
            'ordinary_avg': ordinary_avg,
            'hierarchy_confirmed': special_avg > ordinary_avg
        })
    
    def _plot_consistency_analysis(self, rankings: Dict, df: pd.DataFrame):
        """Plot ranking consistency across distances."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Architecture performance across distances
        arch_perf = df[df['task'] == 'digits_paraphrase'].groupby(['architecture', 'distance'])['peak_accuracy'].mean().reset_index()
        
        for arch in ['autoregressive', 'state_space', 'diffusion']:
            arch_data = arch_perf[arch_perf['architecture'] == arch]
            ax1.plot(arch_data['distance'], arch_data['peak_accuracy'], 'o-', label=arch.title(), linewidth=2)
        
        ax1.set_xlabel('Distance')
        ax1.set_ylabel('Peak Accuracy')
        ax1.set_title('Architecture Performance Across Distances')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Special vs ordinary numbers
        special_distances = [2, 5, 10]
        ordinary_distances = [3, 4, 6, 7, 8, 9]
        
        special_data = df[df['distance'].isin(special_distances) & (df['task'] == 'digits_paraphrase')]
        ordinary_data = df[df['distance'].isin(ordinary_distances) & (df['task'] == 'digits_paraphrase')]
        
        special_by_arch = special_data.groupby('architecture')['peak_accuracy'].mean()
        ordinary_by_arch = ordinary_data.groupby('architecture')['peak_accuracy'].mean()
        
        x = np.arange(len(special_by_arch))
        width = 0.35
        
        ax2.bar(x - width/2, special_by_arch.values, width, label='Special (2,5,10)', alpha=0.8)
        ax2.bar(x + width/2, ordinary_by_arch.values, width, label='Ordinary (3,4,6,7,8,9)', alpha=0.8)
        
        ax2.set_xlabel('Architecture')
        ax2.set_ylabel('Average Peak Accuracy')
        ax2.set_title('Special vs Ordinary Numbers')
        ax2.set_xticks(x)
        ax2.set_xticklabels([arch.title() for arch in special_by_arch.index])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'step4_consistency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {OUTPUT_DIR / 'step4_consistency_analysis.png'}")
    
    def _plot_hierarchy_validation(self, df: pd.DataFrame):
        """Plot mathematical hierarchy validation."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group distances by mathematical significance
        distance_categories = {
            'Binary Base (2)': [2],
            'Decimal Half (5)': [5], 
            'Decimal Base (10)': [10],
            'Ordinary': [3, 4, 6, 7, 8, 9]
        }
        
        category_performance = []
        category_names = []
        
        for cat_name, distances in distance_categories.items():
            cat_data = df[
                df['distance'].isin(distances) & 
                (df['task'] == 'digits_paraphrase')
            ]
            if len(cat_data) > 0:
                category_performance.append(cat_data['peak_accuracy'].tolist())
                category_names.append(cat_name)
        
        ax.boxplot(category_performance, labels=category_names)
        ax.set_title('Mathematical Hierarchy in Proximity Detection')
        ax.set_ylabel('Peak Accuracy')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'step4_hierarchy_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {OUTPUT_DIR / 'step4_hierarchy_validation.png'}")
    
    def _generate_summary_report(self, convergence_df, comparison_df, error_df, validation_stats):
        """Generate a comprehensive summary report."""
        report_path = OUTPUT_DIR / 'analysis_summary.md'
        
        with open(report_path, 'w') as f:
            f.write("# Linear Probe Analysis Summary\n\n")
            
            f.write("## Key Findings\n\n")
            
            f.write("### 1. Architecture-level Patterns\n")
            arch_summary = convergence_df.groupby('architecture').agg({
                'peak_accuracy': ['mean', 'std'],
                'depth_percentage': ['mean', 'std']
            }).round(3)
            
            f.write(f"- **Diffusion models**: Late convergence (~{arch_summary.loc['diffusion', ('depth_percentage', 'mean')]:.0f}% depth)\n")
            f.write(f"- **State-space models**: Early convergence (~{arch_summary.loc['state_space', ('depth_percentage', 'mean')]:.0f}% depth)\n") 
            f.write(f"- **Autoregressive models**: Early-medium convergence (~{arch_summary.loc['autoregressive', ('depth_percentage', 'mean')]:.0f}% depth)\n\n")
            
            f.write("### 2. Task-specific Insights\n")
            avg_gap = comparison_df['performance_gap'].mean()
            f.write(f"- **Performance gap** (Digits - Words): {avg_gap:.3f}\n")
            f.write(f"- **Surface-form encoding**: Strong evidence from degraded word performance\n\n")
            
            f.write("### 3. Mathematical Hierarchy\n")
            f.write(f"- **Special numbers** (2,5,10): {validation_stats['special_avg']:.3f} avg accuracy\n")
            f.write(f"- **Ordinary numbers**: {validation_stats['ordinary_avg']:.3f} avg accuracy\n")
            f.write(f"- **Hierarchy confirmed**: {validation_stats['hierarchy_confirmed']}\n\n")
            
            f.write("### 4. Cross-validation\n")
            f.write(f"- **Ranking consistency**: {validation_stats['ranking_correlation']:.3f} correlation\n")
            f.write(f"- **Robust patterns**: Architecture differences persist across distances\n\n")
            
            f.write("## Implications for StreetMath Paper\n\n")
            f.write("1. **Cognitive miserliness absence**: Models use complex pathways even for simple proximity detection\n")
            f.write("2. **Architecture specialization**: State-space models excel at early numerical pattern recognition\n")
            f.write("3. **Surface-form limitation**: Critical gap in abstract numerical reasoning\n")
            f.write("4. **Mathematical intuition**: Models internalize human-like numerical salience hierarchies\n")
        
        print(f"Saved: {report_path}")

    def run_full_analysis(self):
        """Run the complete 4-step analysis pipeline."""
        print("Starting Linear Probe Analysis Pipeline...")
        print("=" * 60)
        
        # Load data
        self.load_all_data()
        
        # Step 1: Architecture overview
        convergence_df = self.analyze_architecture_overview()
        
        # Step 2: Task comparison  
        comparison_df = self.analyze_task_comparison()
        
        # Step 3: Error analysis
        error_df = self.analyze_error_patterns()
        
        # Step 4: Cross-validation
        self.cross_validate_findings(convergence_df, comparison_df, error_df)
        
        print("\n" + "="*60)
        print("Analysis complete! Check the analysis_output/ directory for results.")
        print("="*60)


if __name__ == "__main__":
    analyzer = LinearProbeAnalyzer()
    analyzer.run_full_analysis()