import os

# 配置
TARGET_KS = [2, 3, 4, 6, 7, 8, 9]

MODELS = [
    ("Qwen/Qwen2.5-3B-Instruct", "Qwen2_5_3B_Instruct"),
    ("Qwen/Qwen3-4B-Thinking-2507", "Qwen3_4B_Thinking"),
    ("Qwen/Qwen2.5-32B-Instruct", "Qwen2_5_32B_Instruct"),
    ("0_Dream-org/0_Dream-v0-Instruct-7B", "Dream_v0_Instruct_7B"),
    ("tiiuae/Falcon-H1-7B-Instruct", "Falcon_H1_7B_Instruct"),
    ("tiiuae/Falcon-H1-34B-Instruct", "Falcon_H1_34B_Instruct"),
]

BASE_DIR = "/Users/pikabp/Repo/StreetMath-v2-main/StreetMathOtherExperiments/LinearProbe"

# ---------------------------------------------------------
# Part 1: Header Template (Contains formatting variables)
# ---------------------------------------------------------
HEADER_TEMPLATE = r'''# %% [markdown]
# # StreetMath Linear Probes — Target: {TARGET_K}
# 
# **Model**: {MODEL_ID}
# **Target**: Near-{TARGET_K} (Threshold: {THRESHOLD})
# 
# This script trains streaming linear probes to detect if a number is "near" a multiple of {TARGET_K}.
# - For k=2,3: Exact match only (Threshold=0).
# - For k>=4: Near match (Threshold=1, i.e., distance 0 or 1).
#

# %% 
import os, re, gc, math, random, json, numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from num2words import num2words

# -----------------------
# Config
# -----------------------
SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MODEL_ID_1 = "{MODEL_ID}"
MODEL_ID_2 = None 

# Dynamic Target Configuration
TARGET_K = {TARGET_K}
NEAR_THRESHOLD = {THRESHOLD}  # 0 for Exact (k=2,3), 1 for Near (k>=4)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

# Data sizes
TRAIN_N = 4000
VAL_N   = 1500

# Efficiency (Conservative defaults for large models)
BATCH_SIZE  = 2
MAX_LEN     = 64
LAYER_STRIDE = 1 

# Output Filename
RESULT_FILE = "probe_results_{SHORT_NAME}_near_{TARGET_K}.json"
'''

# ---------------------------------------------------------
# Part 2: Body Code (Static Python code, no formatting needed)
# ---------------------------------------------------------
BODY_CODE = r'''
# -----------------------
# Data generation (Generalized for K)
# -----------------------
TEMPLATES_A = [
    "Consider the number {n}.",
    "Let x = {n}.",
    "Value: {n}",
    "n: {n}",
    "The integer {n} appears below.",
]
TEMPLATES_B = [
    "Here is {n}.",
    "We study the scalar {n}.",
    "Write down {n} and continue.",
    "Take {n} as the value.",
]

def dist_to_nearest_k(n: int, k: int) -> int:
    rem = n % k
    return min(rem, k - rem)

def rounding_direction(n: int, k: int) -> int:
    # -1 if closer by rounding down, +1 up, 0 exact
    d0 = abs((n // k) * k - n)
    d1 = abs(((n + k - 1) // k) * k - n) # nearest up multiple
    if d0 == 0 or d1 == 0:
        return 0
    return +1 if d1 < d0 else -1


def make_dataset(N: int, templates: List[str], surface: str = "digits", lo: int = 0, hi: int = 9999, seed: int = 1337):
    rng = random.Random(seed)
    texts, labels, metas, spans = [], [], [], []
    for _ in range(N):
        n = rng.randint(lo, hi)
        d = dist_to_nearest_k(n, TARGET_K)
        y = 1 if d <= NEAR_THRESHOLD else 0
        rd = rounding_direction(n, TARGET_K)
        
        if surface == "digits":
            n_str = str(n)
        elif surface == "words":
            n_str = num2words(n).replace('-', ' ').replace(',', '').lower()
        else:
            raise ValueError("surface must be 'digits' or 'words'")
            
        tmpl = rng.choice(templates)
        text = tmpl.format(n=n_str)
        texts.append(text)
        labels.append(y)
        metas.append((d, rd, n))  # (distance, direction, raw n)
    return texts, np.array(labels, dtype=np.int64), metas

# Train on digits + Template A; validate on digits (B) and words (A,B)
train_texts, y_train, meta_train = make_dataset(TRAIN_N, TEMPLATES_A, surface="digits", seed=SEED)
valA_texts,  y_valA,  meta_valA  = make_dataset(VAL_N,   TEMPLATES_B, surface="digits", seed=SEED+1)
valW_texts,  y_valW,  meta_valW  = make_dataset(VAL_N,   TEMPLATES_A, surface="words",  seed=SEED+2)

# -----------------------
# Token span finder
# -----------------------

def find_last_subseq(hay: List[int], needle: List[int]) -> Optional[Tuple[int,int]]:
    if not needle: return None
    found = None
    for i in range(0, len(hay)-len(needle)+1):
        if hay[i:i+len(needle)] == needle:
            found = (i, i+len(needle))
    return found


def pool_number_span(tok, input_ids_row: List[int], layer_h: torch.Tensor, n_str: str) -> np.ndarray:
    n_ids = tok(n_str, add_special_tokens=False)["input_ids"]
    span = find_last_subseq(input_ids_row, n_ids)
    if span is None:
        T = layer_h.shape[0]
        return layer_h[T-1].detach().cpu().numpy()
    s,e = span
    s = min(s, layer_h.shape[0]-1); e = min(e, layer_h.shape[0])
    vec = layer_h[s:e].mean(dim=0)
    return vec.detach().cpu().numpy()

# -----------------------
# Streaming probe pipeline
# -----------------------

def iter_batches(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs], slice(i, min(i+bs, len(lst)))


def model_layer_meta(model, tok) -> Tuple[int,int]:
    with torch.no_grad():
        if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
        tmp = tok("n: 1", return_tensors="pt")
        out = model(**{k:v.to(DEVICE) for k,v in tmp.items()}, use_cache=False, output_hidden_states=True)
        L = len(out.hidden_states)
        D = out.hidden_states[-1].shape[-1]
        del tmp, out
        if DEVICE == "cuda": torch.cuda.empty_cache()
        gc.collect()
        return L, D


def stream_hidden_features(model, tok, texts: List[str], metas: List[Tuple[int,int,int]], layer_ids: List[int], batch_size: int, max_len: int, surface: str) -> Tuple[Dict[int, np.ndarray], np.ndarray, List[Tuple[int,int,int]]]:
    for chunk, s in iter_batches(texts, batch_size):
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        input_ids = enc["input_ids"].to(DEVICE)
        attn      = enc["attention_mask"].to(DEVICE)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False, output_hidden_states=True)
            hs = out.hidden_states
            ids_list = enc["input_ids"].tolist()
            Xb_per_layer = {L: [] for L in layer_ids}
            for b, text in enumerate(chunk):
                if surface == "digits":
                    m = re.findall(r"-?\d+", text)
                    n_str = m[-1] if m else "0"
                else:
                    mm = re.findall(r"[a-z\s]+", text.lower())
                    n_str = max(mm, key=len).strip() if mm else text.lower()
                row_ids = ids_list[b]
                for L in layer_ids:
                    vec = pool_number_span(tok, row_ids, hs[L][b], n_str)
                    Xb_per_layer[L].append(vec)
            for L in layer_ids:
                Xb_per_layer[L] = np.stack(Xb_per_layer[L], axis=0)
        yield Xb_per_layer, s
        del out, hs, enc, input_ids, attn, Xb_per_layer
        if DEVICE == "cuda": torch.cuda.empty_cache()
        gc.collect()


def run_probes_for_model(model_id: str, tag: str, train_pack, eval_packs):
    print(f"\n========== {tag}: {model_id} ==========")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f'Trying slow tokenizer due to error: {e}')
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    
    # Robust Loading (Falcon/0_Dream Style)
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"AutoConfig load failed ({e}); proceeding without explicit config.")
        config = None

    model = None
    load_kwargs = dict(
        torch_dtype=DTYPE,
        output_hidden_states=True,
        trust_remote_code=True,
    )
    if config is not None:
        load_kwargs["config"] = config
    
    try:
        if DEVICE == "cuda":
            # Prefer accelerate's device_map for large models
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", **load_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except TypeError as e:
        print(f"from_pretrained with device_map failed ({e}); retrying without device_map.")
        load_kwargs.pop("config", None) if config is None else None
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, output_hidden_states=True)
        if DEVICE == "cuda":
            model.to(DEVICE)
            
    model.eval(); torch.set_grad_enabled(False)

    total_layers, width = model_layer_meta(model, tok)
    layer_ids = list(range(1, total_layers, LAYER_STRIDE))
    print(f"Layers: {total_layers} | Width: {width} | Probing: {layer_ids[:5]}...")

    # Pass 1: Scalers
    scalers = {L: StandardScaler(with_mean=False) for L in layer_ids}
    texts_tr, y_tr, meta_tr, surf_tr = train_pack
    for Xb_per_layer, s in stream_hidden_features(model, tok, texts_tr, meta_tr, layer_ids, BATCH_SIZE, MAX_LEN, surf_tr):
        for L in layer_ids:
            scalers[L].partial_fit(Xb_per_layer[L])

    # Pass 2: SGD Classifier
    probes = {L: SGDClassifier(loss="log_loss", learning_rate="optimal", alpha=1e-4, tol=None, max_iter=1) for L in layer_ids}
    inited = set()
    for Xb_per_layer, s in stream_hidden_features(model, tok, texts_tr, meta_tr, layer_ids, BATCH_SIZE, MAX_LEN, surf_tr):
        yb = y_tr[s]
        for L in layer_ids:
            Xb = scalers[L].transform(Xb_per_layer[L])
            if L not in inited:
                probes[L].partial_fit(Xb, yb, classes=np.array([0,1], dtype=np.int64))
                inited.add(L)
            else:
                probes[L].partial_fit(Xb, yb)

    # Evaluations
    results = {}
    
    # Dynamic bucket range for error analysis (0 to k//2)
    max_bucket_dist = TARGET_K // 2
    
    for name, (texts_ev, y_ev, meta_ev, surf_ev) in eval_packs.items():
        acc = {L: [] for L in layer_ids}
        for Xb_per_layer, s in stream_hidden_features(model, tok, texts_ev, meta_ev, layer_ids, BATCH_SIZE, MAX_LEN, surf_ev):
            yb = y_ev[s]
            for L in layer_ids:
                Xb = scalers[L].transform(Xb_per_layer[L])
                pb = probes[L].predict(Xb)
                acc[L].extend((pb == yb).tolist())
        
        acc_mean = {L: float(np.mean(acc[L])) if len(acc[L]) else 0.0 for L in layer_ids}
        bestL = max(acc_mean, key=acc_mean.get)
        print(f"Eval[{name}] best layer {bestL} acc={acc_mean[bestL]:.3f}")

        # Detailed Error Analysis on Best Layer
        errs_by_dist = {d: [] for d in range(max_bucket_dist + 1)}
        errs_by_dir  = {{-1:[], 0:[], +1:[]}}
        
        for Xb_per_layer, s in stream_hidden_features(model, tok, texts_ev, meta_ev, [bestL], BATCH_SIZE, MAX_LEN, surf_ev):
            Xb = scalers[bestL].transform(Xb_per_layer[bestL])
            yb = y_ev[s]
            proba = probes[bestL].predict_proba(Xb)[:,1]
            preds = (proba >= 0.5).astype(np.int64)
            for i, idx in enumerate(range(s.start, s.stop)):
                d, rd, n = meta_ev[idx]
                d_bucket = min(d, max_bucket_dist)
                if d_bucket in errs_by_dist:
                    errs_by_dist[d_bucket].append(int(preds[i] != yb[i]))
                if rd in errs_by_dir:
                    errs_by_dir[rd].append(int(preds[i] != yb[i]))
                    
        dist_view = {k: (float(np.mean(v)) if len(v) else None, len(v)) for k,v in errs_by_dist.items()}
        dir_view  = {k: (float(np.mean(v)) if len(v) else None, len(v)) for k,v in errs_by_dir.items()}
        results[name] = {"acc_per_layer": acc_mean, "best_layer": bestL, "err_by_dist": dist_view, "err_by_dir": dir_view}

    del model, tok
    if DEVICE=="cuda": torch.cuda.empty_cache()
    gc.collect()
    return results

# -----------------------
# Run
# -----------------------
all_results = {}
all_results["model1"] = run_probes_for_model(MODEL_ID_1, tag="MODEL_1", train_pack=train_pack, eval_packs=eval_packs)

print(f"Saving results to {RESULT_FILE}...")
with open(RESULT_FILE, "w") as f:
    json.dump(all_results, f, indent=2)
print("Done.")
'''

def main():
    if not os.path.exists(BASE_DIR):
        print(f"Error: Base dir {BASE_DIR} does not exist.")
        return

    for k in TARGET_KS:
        # 1. Create directory
        dir_name = f"near_{k}"
        dir_path = os.path.join(BASE_DIR, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created/Checked directory: {dir_path}")

        # Determine threshold logic
        threshold = 0 if k < 4 else 1

        for model_id, short_name in MODELS:
            # 2. Format Header
            header = HEADER_TEMPLATE.format(
                TARGET_K=k,
                THRESHOLD=threshold,
                MODEL_ID=model_id,
                SHORT_NAME=short_name
            )
            
            # 3. Concatenate
            content = header + BODY_CODE
            
            # 4. Write file
            filename = f"linear_probe_{short_name}.py"
            file_path = os.path.join(dir_path, filename)
            with open(file_path, "w") as f:
                f.write(content)
            print(f"  -> Generated {filename}")

if __name__ == "__main__":
    main()