import sys
import os

# 把当前目录加入 path，以便 import
sys.path.append(os.getcwd())

# 导入目标脚本作为一个模块 (Falcon 34B Near-7)
import linear_probe_falcon_34b_near_7 as probe_script

print("=== STARTING DRY RUN FOR FALCON 34B NEAR-7 ===")

# 1. 动态修改配置
probe_script.TRAIN_N = 20  # Increase to avoid NaN in scaler
probe_script.VAL_N   = 5
probe_script.BATCH_SIZE = 1 # 34B 必须用 1

print(f"Patched Config: TRAIN_N={probe_script.TRAIN_N}, BATCH_SIZE={probe_script.BATCH_SIZE}")

# 2. 重新生成数据 (覆盖大 N 数据)
print("Regenerating dataset with small size...")
train_texts, y_train, meta_train = probe_script.make_dataset(
    probe_script.TRAIN_N, probe_script.TEMPLATES_A, surface="digits", seed=probe_script.SEED
)
valA_texts, y_valA, meta_valA = probe_script.make_dataset(
    probe_script.VAL_N, probe_script.TEMPLATES_B, surface="digits", seed=probe_script.SEED+1
)
valW_texts, y_valW, meta_valW = probe_script.make_dataset(
    probe_script.VAL_N, probe_script.TEMPLATES_A, surface="words", seed=probe_script.SEED+2
)

# 3. 重新组装 Packs
train_pack = (train_texts, y_train, meta_train, "digits")
eval_packs = {
    "digits_paraphrase": (valA_texts, y_valA, meta_valA, "digits"),
    "words":             (valW_texts, y_valW, meta_valW, "words"),
}

# 4. 运行主逻辑
print("Running probe logic (this will download model if not cached)...")
try:
    results = probe_script.run_probes_for_model(
        probe_script.MODEL_ID_1, 
        tag="DRY_RUN_FALCON_34B", 
        train_pack=train_pack,  
        eval_packs=eval_packs   
    )
    print("\n=== DRY RUN SUCCESS ===")
    print("Results layers detected:", len(results["digits_paraphrase"]["acc_per_layer"]))
    
except Exception as e:
    print(f"\n=== DRY RUN FAILED ===")
    print(e)
    import traceback
    traceback.print_exc()
