#!/usr/bin/env python3
"""
Plot Linear Probe Accuracy by Distance
=====================================

Generate line plots showing model performance across different near distances.
- X-axis: Near distance (2-10)
- Y-axis: Average accuracy (weighted by sample count from err_by_dist)
- Each line represents a different model
- Consistent with plot_acc_per_layer.py colors and styling
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


def load_model_mapping(mapping_file: Path) -> dict:
    """从mapping.txt加载模型名称映射"""
    mapping = {}
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line:
                    key, value = line.split(' = ', 1)
                    mapping[key.strip()] = value.strip()
    return mapping


def get_mean_accuracy(acc_per_layer: Dict) -> float:
    """Get the mean accuracy across all layers."""
    if not acc_per_layer:
        return 0.0
    accuracies = [float(acc) for acc in acc_per_layer.values()]
    return sum(accuracies) / len(accuracies)


def load_accuracy_data(base_dir: Path, task: str) -> Dict[str, Dict[int, float]]:
    """
    Load mean accuracy data for all models across all distances.
    Returns: {model_key: {distance: mean_accuracy}}
    """
    distances = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    model_data = {}
    
    for distance in distances:
        distance_dir = base_dir / str(distance)
        if not distance_dir.exists():
            continue
        
        for json_file in distance_dir.glob("probe_results_*.json"):
            # Extract model name from filename
            model_key = json_file.stem.replace('probe_results_', '').replace(f'_near_{distance}', '')
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract task data
            model1_data = data.get('model1', {})
            task_data = model1_data.get(task, {})
            acc_per_layer = task_data.get('acc_per_layer', {})
            
            if acc_per_layer:
                mean_acc = get_mean_accuracy(acc_per_layer)
                
                if model_key not in model_data:
                    model_data[model_key] = {}
                model_data[model_key][distance] = mean_acc
    
    return model_data


def plot_accuracy_by_distance():
    """Generate line plots for both digits and words tasks."""
    base_dir = Path(__file__).resolve().parent / "results"
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(exist_ok=True)
    
    # Load model mapping
    mapping_file = Path(__file__).resolve().parent / "mapping.txt"
    model_mapping = load_model_mapping(mapping_file)
    
    # Define consistent model order and colors (same as plot_acc_per_layer.py)
    model_order = [
        'dream',
        'falcon_h1_7b_instruct', 
        'falcon_h1_34b_instruct',
        'qwen_2_5_3b_instruct',
        'qwen_2_5_32b_instruct'
    ]
    
    # Same colors as plot_acc_per_layer.py
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    model_colors = {model: colors[i] for i, model in enumerate(model_order)}
    
    distances = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Generate plots for both tasks
    tasks = [
        ('digits_paraphrase', 'Digits'),
        ('words', 'Words')
    ]
    
    for task_key, task_name in tasks:
        print(f"Processing {task_name} task...")
        
        # Load data for this task
        model_accuracies = load_accuracy_data(base_dir, task_key)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot each model
        for model_key in model_order:
            if model_key not in model_accuracies:
                continue
            
            model_data = model_accuracies[model_key]
            
            # Extract x and y values for available distances
            x_vals = []
            y_vals = []
            
            for distance in distances:
                if distance in model_data:
                    x_vals.append(distance)
                    y_vals.append(model_data[distance])
            
            if x_vals and y_vals:
                # Get model display name and color
                display_name = model_mapping.get(model_key, model_key)
                color = model_colors[model_key]
                
                # Plot line
                plt.plot(x_vals, y_vals, 
                        marker='o', 
                        linewidth=2.0, 
                        markersize=6, 
                        color=color,
                        label=display_name)
        
        # Formatting
        plt.xlabel('Near Distance', fontsize=12)
        plt.ylabel('Mean Accuracy Across Layers', fontsize=12)
        plt.title(f'Linear Probe Mean Accuracy by Distance: {task_name}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Set x-axis ticks to show all distances
        plt.xticks(distances)
        
        # Set y-axis limits for better visualization
        plt.ylim(0, 1.05)
        
        # Legend configuration (same style as plot_acc_per_layer.py)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)
        
        # Layout adjustment
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.20)
        
        # Save figure
        output_filename = f'accuracy_by_number_{task_key}.png'
        output_path = output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        
        plt.close()


def main():
    """Main function."""
    print("Generating Linear Probe accuracy by distance plots...")
    plot_accuracy_by_distance()
    print("Done!")


if __name__ == '__main__':
    main()