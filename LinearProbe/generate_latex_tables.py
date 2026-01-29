#!/usr/bin/env python3
"""
Generate LaTeX tables for Linear Probe results for near-2 through near-10 experiments.
"""

import json
import os
from pathlib import Path

def load_json_data(file_path):
    """Load and return JSON data from file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get('model1', {})
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in file: {file_path}")
        return None

def get_peak_accuracy(acc_per_layer):
    """Get the maximum accuracy from acc_per_layer dictionary."""
    if not acc_per_layer:
        return 0.0
    return max(float(acc) for acc in acc_per_layer.values())

def format_error_rate(error_rate):
    """Format error rate as percentage with 1 decimal place."""
    return f"{error_rate * 100:.1f}\\%"

def load_model_mapping(mapping_file):
    """Load model mapping from mapping.txt file."""
    mapping = {}
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line:
                    key, value = line.split(' = ', 1)
                    mapping[key.strip()] = value.strip()
    return mapping

def get_model_total_layers(model_key):
    """Get total number of layers for each model."""
    layer_info = {
        'dream': 28,
        'falcon_h1_7b_instruct': 44, 
        'falcon_h1_34b_instruct': 72,
        'qwen_2_5_3b_instruct': 36,
        'qwen_2_5_32b_instruct': 64,
        'qwen3_4b_thinking': 36
    }
    return layer_info.get(model_key, None)

def get_model_display_name(model_key, mapping=None):
    """Convert model key to display name using mapping file."""
    if mapping is None:
        # Fallback mapping if file not found
        mapping = {
            'falcon_h1_7b_instruct': 'Falcon-H1-7B-Instruct',
            'falcon_h1_34b_instruct': 'Falcon-H1-34B-Instruct',
            'qwen_2_5_3b_instruct': 'Qwen2.5-3B-Instruct',
            'qwen_2_5_32b_instruct': 'Qwen2.5-32B-Instruct',
            'qwen3_4b_thinking': 'Qwen3-4B-Thinking',
            'dream': 'Dream-7B'
        }
    return mapping.get(model_key, model_key)

def get_distance_mapping():
    """Get mapping of target numbers to distance keys (including distance 0)."""
    return {
        2: [0, 1],
        3: [0, 1], 
        4: [0, 1, 2],
        5: [0, 1, 2],
        6: [0, 1, 2, 3],
        7: [0, 1, 2, 3],
        8: [0, 1, 2, 3, 4],
        9: [0, 1, 2, 3, 4],
        10: [0, 1, 2, 3, 4, 5]
    }

def generate_latex_table(target_num, format_type, model_results, model_mapping=None):
    """Generate LaTeX table content for given target number and format."""
    
    # Get distance keys for this target (including distance 0)
    distance_mapping = get_distance_mapping()
    distance_keys = distance_mapping.get(target_num, [0, 1])
    
    # Determine table caption and label
    if format_type == 'digits':
        caption = f"Comprehensive Near-{target_num} Digit Analysis: Performance and Error Patterns at the best layer. Acc = Accuracy; Err = Error rate"
        label = f"near{target_num}"
        title_suffix = ""
    else:
        caption = f"Comprehensive Near-{target_num} (Words) Analysis: Performance and Error Patterns at the best layer. Acc = Accuracy; Err = Error rate"
        label = f"near{target_num}_words"
        title_suffix = " (Words)"
    
    # Calculate number of columns: Model + Peak Acc + Total Layers + Best Layer + Error columns (including distance 0)
    num_error_cols = len(distance_keys)
    total_cols = 4 + num_error_cols  # 4 for Model, Peak Acc, Total Layers, Best Layer
    
    # Adjust column width for model name based on number of columns
    if total_cols > 5:
        model_width = "0.25\\textwidth"
        tab_sep = "1.0pt"
    else:
        model_width = "0.3\\textwidth"
        tab_sep = "1.5pt"
    
    # Build column specification: Model column + fixed number of centered columns  
    col_spec = f"|>{{\\raggedright\\arraybackslash}}p{{{model_width}}}|" + "c|" * (total_cols - 1)
    
    # Start building the LaTeX table
    latex_content = []
    latex_content.append("\\begin{table*}[!t]")
    latex_content.append("  \\centering")
    latex_content.append("  \\small")
    latex_content.append(f"  \\setlength{{\\tabcolsep}}{{{tab_sep}}}")
    latex_content.append("  \\renewcommand{\\arraystretch}{1.05}")
    latex_content.append(f"  \\caption{{{caption}}}")
    latex_content.append(f"  \\label{{tab:{label}}}")
    latex_content.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    latex_content.append("  \\hline")
    
    # Build header row dynamically
    header_parts = ["\\textbf{Model}", "\\textbf{Peak Acc}", "\\textbf{Total Layers}", "\\textbf{Best Layer}"]
    for dist in distance_keys:
        header_parts.append(f"\\textbf{{Err ({dist})}}")
    
    header = "  " + " & ".join(header_parts) + " \\\\"
    latex_content.append(header)
    latex_content.append("  \\hline")
    
    # Sort models by display name for consistent ordering
    sorted_models = sorted(model_results.items(), key=lambda x: get_model_display_name(x[0], model_mapping))
    
    for model_key, model_data in sorted_models:
        if model_data is None:
            continue
            
        display_name = get_model_display_name(model_key, model_mapping)
        
        # Get data for the specified format
        format_key = 'digits_paraphrase' if format_type == 'digits' else 'words'
        data = model_data.get(format_key, {})
        
        if not data:
            continue
            
        # Extract metrics
        peak_acc = get_peak_accuracy(data.get('acc_per_layer', {}))
        best_layer = data.get('best_layer', 0)
        err_by_dist = data.get('err_by_dist', {})
        
        # Get layer position information
        total_layers = get_model_total_layers(model_key)
        if total_layers and best_layer > 0:
            layer_percentage = (best_layer / total_layers) * 100
            best_layer_info = f"{best_layer} ({layer_percentage:.0f}\\% depth)"
        else:
            best_layer_info = str(best_layer)
        
        total_layers_info = str(total_layers) if total_layers else "N/A"
        
        # Get error rates for each distance (including distance 0)
        error_values = []
        for dist in distance_keys:
            dist_str = str(dist)
            if dist_str in err_by_dist:
                error_rate = err_by_dist[dist_str][0]
                error_values.append(format_error_rate(error_rate))
            else:
                error_values.append("N/A")
        
        # Format the row
        if len(display_name) > 20:  # Use makecell for long names
            name_cell = f"\\makecell[tl]{{{display_name}}}"
        else:
            name_cell = display_name
        
        # Build row with dynamic error columns
        row_parts = [name_cell, f"{peak_acc:.3f}", total_layers_info, best_layer_info] + error_values
        row = "  " + " & ".join(row_parts) + " \\\\"
        latex_content.append(row)
        latex_content.append("  \\hline")
    
    latex_content.append("  \\end{tabular}")
    latex_content.append("\\end{table*}")
    
    return "\n".join(latex_content)

def main():
    """Main function to generate all LaTeX tables."""
    base_dir = "/Users/pikabp/Repo/StreetMath-v2/LinearProbe/results"
    output_dir = "/Users/pikabp/Repo/StreetMath-v2/LinearProbe/tables"
    
    # Load model mapping from mapping.txt (in parent directory)
    mapping_file = os.path.join(os.path.dirname(base_dir), "mapping.txt")
    model_mapping = load_model_mapping(mapping_file)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Model file patterns (excluding qwen3_4b_thinking due to hint prompt conflicts)
    model_files = [
        'falcon_h1_7b_instruct',
        'falcon_h1_34b_instruct', 
        'qwen_2_5_3b_instruct',
        'qwen_2_5_32b_instruct',
        'dream'
    ]
    
    # Process each target number from 2 to 10
    for target_num in range(2, 11):
        print(f"Processing near-{target_num}...")
        
        # Directory name is just the target number without leading zeros
        dir_name = str(target_num)
        
        target_dir = os.path.join(base_dir, dir_name)
        
        # Collect data from all models
        model_results = {}
        for model_key in model_files:
            file_name = f"probe_results_{model_key}_near_{target_num}.json"
            file_path = os.path.join(target_dir, file_name)
            model_results[model_key] = load_json_data(file_path)
        
        # Generate both digits and words tables
        for format_type in ['digits', 'words']:
            table_content = generate_latex_table(target_num, format_type, model_results, model_mapping)
            
            # Determine output filename
            if format_type == 'digits':
                output_file = f"Linear_probe_near_{target_num}.tex"
            else:
                output_file = f"Linear_probe_near_{target_num}_words.tex"
            
            output_path = os.path.join(output_dir, output_file)
            
            # Write the table to file
            with open(output_path, 'w') as f:
                f.write(table_content)
            
            print(f"Generated: {output_path}")

if __name__ == "__main__":
    main()