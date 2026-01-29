import json
from pathlib import Path
import matplotlib.pyplot as plt


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


def load_acc_per_layer(json_path: Path, task_key: str):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Expected structure: top -> model key -> task key -> 'acc_per_layer'
    # Find first top-level dict value that contains the specified task_key
    acc_per_layer = None

    for _, model_val in data.items():
        if isinstance(model_val, dict) and task_key in model_val:
            task = model_val[task_key]
            if isinstance(task, dict) and isinstance(task.get('acc_per_layer'), dict):
                acc_per_layer = task['acc_per_layer']
                break

    if acc_per_layer is None:
        raise ValueError(f"Task '{task_key}' with acc_per_layer not found in {json_path}")

    # Convert string keys to ints and sort
    items = sorted(((int(k), v) for k, v in acc_per_layer.items()), key=lambda kv: kv[0])
    layers = [k for k, _ in items]
    accs = [v for _, v in items]
    return layers, accs


def plot_for_task(task_key: str, near_tag: str, out_name: str):
    base = Path(__file__).resolve().parent
    # 加载模型名称映射
    model_mapping = load_model_mapping(base / 'mapping.txt')
    
    # 定义固定的模型顺序和颜色映射 (excluding qwen3_4b_thinking due to hint prompt conflicts)
    model_order = [
        'dream',
        'falcon_h1_7b_instruct', 
        'falcon_h1_34b_instruct',
        'qwen_2_5_3b_instruct',
        'qwen_2_5_32b_instruct'
    ]
    
    # 定义固定的颜色映射（使用matplotlib的默认颜色循环）
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    model_colors = {model: colors[i] for i, model in enumerate(model_order)}
    
    # 新的目录列表
    distance_dirs = ['2', '3', '4', '5', '6', '7', '8', '9', '10']

    # 收集文件并按模型组织
    model_files = {}
    for dist_dir in distance_dirs:
        if near_tag in f'near_{dist_dir}':  # 匹配对应的距离
            dir_path = base / 'results' / dist_dir
            if dir_path.exists():
                # 找到该目录下所有的 probe_results_*.json 文件
                json_files = list(dir_path.glob('probe_results_*.json'))
                for p in json_files:
                    raw_model_name = p.stem.replace('probe_results_', '').replace(f'_near_{p.parent.name}', '')
                    if raw_model_name in model_order:
                        model_files[raw_model_name] = p

    if not model_files:
        raise RuntimeError(f"No files matched near tag '{near_tag}'")

    plt.figure(figsize=(16, 6))

    # 按固定顺序绘制模型，确保颜色和图例顺序一致
    for model in model_order:
        if model in model_files:
            p = model_files[model]
            if not p.exists():
                raise FileNotFoundError(f"Missing expected file: {p}")
            layers, accs = load_acc_per_layer(p, task_key)
            label = model_mapping.get(model, model)  # 如果找不到映射就用原名
            color = model_colors[model]
            plt.plot(layers, accs, marker='o', linewidth=1.5, markersize=3, 
                    label=label, color=color)

    plt.xlabel('Layer/Step Number')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy per Layer across Probes ({task_key}, {near_tag})')
    plt.grid(True, alpha=0.3)
    # Place legend at the bottom
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    # Fit layout and then expand bottom margin so legend doesn't overlap
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.32)

    out_path = base / 'figures' / out_name
    plt.savefig(out_path, dpi=200)
    print(f"Saved figure to: {out_path}")


def main():
    # Create four plots: task x distance (near_5, near_10)
    distances = ['2', '3', '4', '5', '6', '7', '8', '9', '10']
    for dist in distances:
        plot_for_task('digits_paraphrase', f'near_{dist}', f'acc_per_layer_digits_paraphrase_near_{dist}.png')
        plot_for_task('words', f'near_{dist}', f'acc_per_layer_words_near_{dist}.png')

if __name__ == '__main__':
    main()
