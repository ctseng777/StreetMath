import sys
import os

# 把当前目录加入 path，以便 import
sys.path.append(os.getcwd())

# 导入目标脚本作为一个模块 (你可以改这个名字来测不同的脚本)
import linear_probe_falcon_7b_near_7 as probe_script

print("=== STARTING DRY RUN FOR FALCON 7B NEAR-7 ===")

# 1. 动态修改配置
probe_script.TRAIN_N = 20  # 极小数据量
probe_script.VAL_N   = 10
probe_script.BATCH_SIZE = 2

print(f"Patched Config: TRAIN_N={probe_script.TRAIN_N}, VAL_N={probe_script.VAL_N}")

# 2. 重新生成数据 (这是关键！必须手动调用以覆盖原模块加载时生成的大数据)
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

# 3. 重新组装 Packs (这一步在原脚本里是写死的，所以我们要在 dry run 里重组)
train_pack = (train_texts, y_train, meta_train, "digits")
eval_packs = {
    "digits_paraphrase": (valA_texts, y_valA, meta_valA, "digits"),
    "words":             (valW_texts, y_valW, meta_valW, "words"),
}

# 4. 运行主逻辑
print("Running probe logic (this will try to load the model)...")
try:
    # 这里的 run_probes_for_model 会加载模型。
    # 如果你在本地 Mac 且没有 GPU/内存不够，这里可能会卡住或报错。
    # 但这能验证代码逻辑是否通顺。
    results = probe_script.run_probes_for_model(
        probe_script.MODEL_ID_1, 
        tag="DRY_RUN", 
        train_pack=train_pack,  # 传入我们要的小数据包
        eval_packs=eval_packs   # 传入我们要的小数据包
    )
    print("\n=== DRY RUN SUCCESS ===")
    print("Results sample:", list(results.keys()))
    
except Exception as e:
    print(f"\n=== DRY RUN FAILED ===")
    print(e)
    # 打印详细堆栈
    import traceback
    traceback.print_exc()