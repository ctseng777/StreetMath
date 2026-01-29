# %% [markdown]
# # StreetMath Linear Probes (FALCON 34B) — Target: 7
# 
# **Model**: tiiuae/Falcon-H1-34B-Instruct
# **Target**: Near-7 (Threshold: 1)
# 
# Uses robust loading logic for Falcon architectures.
# **Optimized for 34B**: Batch Size = 1.
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

MODEL_ID_1 = "tiiuae/Falcon-H1-34B-Instruct"
MODEL_ID_2 = None 

TARGET_K = 10
NEAR_THRESHOLD = 1 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Efficiency (Ultra-conservative for 34B)
TRAIN_N = 4000
VAL_N   = 1500
BATCH_SIZE  = 1  # Crucial for 34B
MAX_LEN     = 48 # Reduced length
LAYER_STRIDE = 1 

RESULT_FILE = "probe_results_falcon_h1_34b_instruct_near_10.json"

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
    d0 = abs((n // k) * k - n)
    d1 = abs(((n + k - 1) // k) * k - n)
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
        metas.append((d, rd, n))
    return texts, np.array(labels, dtype=np.int64), metas

train_texts, y_train, meta_train = make_dataset(TRAIN_N, TEMPLATES_A, surface="digits", seed=SEED)
valA_texts,  y_valA,  meta_valA  = make_dataset(VAL_N,   TEMPLATES_B, surface="digits", seed=SEED+1)
valW_texts,  y_valW,  meta_valW  = make_dataset(VAL_N,   TEMPLATES_A, surface="words",  seed=SEED+2)

# Pack data (Global variables)
train_pack = (train_texts, y_train, meta_train, "digits")
eval_packs = {
    "digits_paraphrase": (valA_texts, y_valA, meta_valA, "digits"),
    "words":             (valW_texts, y_valW, meta_valW, "words"),
}

# ----------------------- 
# Helpers
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
        return layer_h[T-1].detach().float().cpu().numpy()
    s,e = span
    s = min(s, layer_h.shape[0]-1); e = min(e, layer_h.shape[0])
    vec = layer_h[s:e].mean(dim=0)
    return vec.detach().float().cpu().numpy()

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
    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"  [Progress] Starting stream for {len(texts)} samples ({total_batches} batches)...")
    
    total_nan_count = 0
    total_elements = 0
    
    for batch_idx, (chunk, s) in enumerate(iter_batches(texts, batch_size)):
        log_freq = 100 if batch_size == 1 else 10
        if (batch_idx + 1) % log_freq == 0:
            nan_rate = (total_nan_count / total_elements * 100) if total_elements > 0 else 0
            print(f"    Processing batch {batch_idx + 1}/{total_batches}... (Total NaN found: {total_nan_count}, Rate: {nan_rate:.4f}%)", flush=True)
            
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
                    m = re.findall(r"-?\d+", text); n_str = m[-1] if m else "0"
                else:
                    mm = re.findall(r"[a-z\s]+", text.lower()); n_str = max(mm, key=len).strip() if mm else text.lower()
                row_ids = ids_list[b]
                for L in layer_ids:
                    vec = pool_number_span(tok, row_ids, hs[L][b], n_str)
                    # Force float32
                    vec = vec.astype(np.float32)
                    
                    # Count NaNs/Infs before cleaning
                    nans = np.sum(~np.isfinite(vec))
                    total_nan_count += nans
                    total_elements += vec.size
                    
                    # Sanitize
                    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
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
    
    # Robust Loading
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.output_hidden_states = True # Set explicitly on config
    except Exception as e:
        print(f"AutoConfig load failed ({e}); proceeding without config.")
        config = None

    load_kwargs = dict(torch_dtype=DTYPE, trust_remote_code=True) # Removed output_hidden_states here
    if config: load_kwargs["config"] = config
    
    try:
        if DEVICE == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                device_map="auto", 
                low_cpu_mem_usage=True,
                offload_folder="/workspace/offload",
                offload_state_dict=True,
                **load_kwargs
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except TypeError:
        print("Device map failed, retrying without.")
        load_kwargs.pop("config", None) if config is None else None
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, output_hidden_states=True)
        if DEVICE == "cuda": model.to(DEVICE)
            
    model.eval(); torch.set_grad_enabled(False)

    total_layers, width = model_layer_meta(model, tok)
    layer_ids = list(range(1, total_layers, LAYER_STRIDE))
    print(f"Layers: {total_layers} | Width: {width}")

    scalers = {L: StandardScaler(with_mean=False) for L in layer_ids}
    texts_tr, y_tr, meta_tr, surf_tr = train_pack
    for Xb_per_layer, s in stream_hidden_features(model, tok, texts_tr, meta_tr, layer_ids, BATCH_SIZE, MAX_LEN, surf_tr):
        for L in layer_ids:
            scalers[L].partial_fit(Xb_per_layer[L])

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

    results = {}
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

        errs_by_dist = {d: [] for d in range(max_bucket_dist + 1)}
        errs_by_dir  = {-1:[], 0:[], +1:[]}
        
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

if __name__ == "__main__":
    all_results = {}
    all_results["model1"] = run_probes_for_model(MODEL_ID_1, tag="FALCON_34B", train_pack=train_pack, eval_packs=eval_packs)

    with open(RESULT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)