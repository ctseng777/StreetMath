import json
import os
import numpy as np

# 你的 Falcon 7B 结果文件所在的目录
TARGET_DIR = "/Users/pikabp/Repo/StreetMath-v2-main/StreetMathOtherExperiments/LinearProbe_v1/0_Falcon-H1-7B-Instruct/"

def fix_json_structure(file_path):
    print(f"Fixing {file_path}...")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 遍历 models (通常是 "model1")
        for model_key, model_data in data.items():
            # 遍历 splits (例如 "digits_paraphrase", "words")
            for split_name, split_data in model_data.items():
                if not isinstance(split_data, dict): continue
                
                # 1. Fix err_by_dist
                if "err_by_dist" in split_data:
                    new_dist = {}
                    for k, v in split_data["err_by_dist"].items():
                        # 如果已经是 [mean, count] 格式 (list 长度为2且第一个是浮点或None)，跳过
                        # 但为了保险，我们检查 v 是否是原始的长列表
                        if isinstance(v, list) and len(v) > 2: 
                            # 这是一个原始列表 [0, 1, 0, ...]
                            count = len(v)
                            mean_val = float(np.mean(v)) if count > 0 else None
                            new_dist[k] = (mean_val, count)
                        else:
                            # 已经是修复好的，或者是空列表
                            new_dist[k] = v
                    split_data["err_by_dist"] = new_dist

                # 2. Fix err_by_dir
                if "err_by_dir" in split_data:
                    new_dir = {}
                    for k, v in split_data["err_by_dir"].items():
                        if isinstance(v, list) and len(v) > 2:
                            count = len(v)
                            mean_val = float(np.mean(v)) if count > 0 else None
                            new_dir[k] = (mean_val, count)
                        else:
                            new_dir[k] = v
                    split_data["err_by_dir"] = new_dir
        
        # 覆盖保存
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print("  -> Done.")
        
    except Exception as e:
        print(f"  -> Error: {e}")

if __name__ == "__main__":
    # 找到所有 probe_results_*.json
    files = [f for f in os.listdir(TARGET_DIR) if f.startswith("probe_results_") and f.endswith(".json")]
    
    if not files:
        print(f"No JSON files found in {TARGET_DIR}")
    else:
        for f in files:
            fix_json_structure(os.path.join(TARGET_DIR, f))
