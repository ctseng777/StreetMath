import sys
import os

sys.path.append(os.getcwd())

import linear_probe_qwen_32b_near_7 as probe_script

print("=== STARTING DRY RUN FOR QWEN 32B NEAR-7 ===")

# 1. Monkey Patch
probe_script.TRAIN_N = 20
probe_script.VAL_N   = 10
probe_script.BATCH_SIZE = 1  # Crucial for 32B

print(f"Patched Config: TRAIN_N={probe_script.TRAIN_N}, BATCH_SIZE={probe_script.BATCH_SIZE}")

# 2. Regenerate Data
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

# 3. Repack
train_pack = (train_texts, y_train, meta_train, "digits")
eval_packs = {
    "digits_paraphrase": (valA_texts, y_valA, meta_valA, "digits"),
    "words":             (valW_texts, y_valW, meta_valW, "words"),
}

# 4. Run
print("Running probe logic...")
try:
    results = probe_script.run_probes_for_model(
        probe_script.MODEL_ID_1, 
        tag="DRY_RUN_QWEN_32B", 
        train_pack=train_pack,  
        eval_packs=eval_packs   
    )
    print("\n=== DRY RUN SUCCESS ===")
    print("Results layers detected:", len(results["digits_paraphrase"]["acc_per_layer"]))
    
except Exception as e:
    print(f"\n=== DRY RUN FAILED ===")
    import traceback
    traceback.print_exc()
