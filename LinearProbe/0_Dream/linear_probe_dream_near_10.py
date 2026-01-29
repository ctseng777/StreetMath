#!/usr/bin/env python3
"""
Linear probe script for Dream-v0-Instruct-7B diffusion language model (Near-10 Task).
Captures hidden states at every layer during forward pass to detect proximity to multiples of 10.
"""

import os, re, gc, math, random, json, numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from num2words import num2words
import warnings
warnings.filterwarnings('ignore')

# -----------------------
# Config (tune here)
# -----------------------
SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Dream model configuration
MODEL_ID_1 = "Dream-org/Dream-v0-Instruct-7B"  # Dream diffusion LLM
MODEL_ID_2 = None  # Optional: comparison model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

# Data sizes
TRAIN_N = 2000
VAL_N   = 500

# Efficiency
BATCH_SIZE  = 16
MAX_LEN     = 48
LAYER_STRIDE = 1

# Probing target: near‑10 = distance to nearest multiple of 10 <= 1
TARGET_K = 10
NEAR_THRESHOLD = 1

RESULT_FILE = "probe_results_dream_near_10.json"

# -----------------------
# Data generation
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

# Pre-generate datasets
train_texts, y_train, meta_train = make_dataset(TRAIN_N, TEMPLATES_A, surface="digits", seed=SEED)
valA_texts,  y_valA,  meta_valA  = make_dataset(VAL_N,   TEMPLATES_B, surface="digits", seed=SEED+1)
valW_texts,  y_valW,  meta_valW  = make_dataset(VAL_N,   TEMPLATES_A, surface="words",  seed=SEED+2)

# Pack data
train_pack = (train_texts, y_train, meta_train, "digits")
eval_packs = {
    "digits_paraphrase": (valA_texts, y_valA, meta_valA, "digits"),
    "words":             (valW_texts, y_valW, meta_valW, "words"),
}

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
    s, e = span
    s = int(min(s, layer_h.shape[0]-1))
    e = int(min(e, layer_h.shape[0]))
    vec = layer_h[s:e].mean(dim=0)
    return vec.detach().cpu().numpy()

# -----------------------
# Model loading for Dream architecture
# -----------------------
def load_dream_model(model_id: str):
    """Load Dream diffusion language model with proper configuration."""
    print(f"Loading Dream model: {model_id}...")
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        if DEVICE == "cuda":
            model = model.to(DEVICE)
        return model, "dream"
    except Exception as e:
        print(f"Warning: Error loading with AutoModelForCausalLM: {e}")
        print("Attempting fallback to AutoModel...")
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_id, torch_dtype=DTYPE, trust_remote_code=True)
        if DEVICE == "cuda":
            model = model.to(DEVICE)
        return model, "auto"

# -----------------------
# Streaming helpers
# -----------------------
def iter_batches(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs], slice(i, min(i+bs, len(lst)))

def model_layer_meta(model, model_type: str) -> Tuple[int,int]:
    with torch.no_grad():
        tok_path = MODEL_ID_1
        tmp_tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        if tmp_tok.pad_token_id is None: 
            tmp_tok.pad_token_id = tmp_tok.eos_token_id if tmp_tok.eos_token_id is not None else 0
        test_text = "The number 42"
        enc = tmp_tok(test_text, return_tensors="pt", padding=False, truncation=True, max_length=16)
        inputs = {k: v.to(DEVICE) for k, v in enc.items() if torch.is_tensor(v)}
        try:
            out = model(**inputs, use_cache=False, output_hidden_states=True, return_dict=True)
            L = len(out.hidden_states)
            D = out.hidden_states[-1].shape[-1]
        except Exception as e:
            print(f"Warning: Error getting layer meta: {e}")
            L = 29; D = 3584
        del enc, inputs
        if DEVICE == "cuda": torch.cuda.empty_cache()
        gc.collect()
        return L, D

def stream_hidden_features(model, model_type, tok, texts, metas, layer_ids, batch_size, max_len, surface):
    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"  [Progress] Starting stream for {len(texts)} samples ({total_batches} batches)...")
    for batch_idx, (chunk, s) in enumerate(iter_batches(texts, batch_size)):
        if (batch_idx + 1) % 10 == 0:
            print(f"    Processing batch {batch_idx + 1}/{total_batches}...", flush=True)
        enc = tok(chunk, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
        input_ids = enc["input_ids"].to(DEVICE)
        if input_ids.size(0) == 0: continue
        with torch.no_grad():
            try:
                outputs = model(input_ids=input_ids, use_cache=False, output_hidden_states=True, return_dict=True)
                hs = outputs.hidden_states
            except: continue
            ids_list = enc["input_ids"].tolist()
            Xb_per_layer = {L: [] for L in layer_ids}
            for b, text in enumerate(chunk):
                if surface == "digits":
                    m = re.findall(r"-?\d+", text); n_str = m[-1] if m else "0"
                else:
                    mm = re.findall(r"[a-z\s]+", text.lower()); n_str = max(mm, key=len).strip() if mm else text.lower()
                row_ids = ids_list[b]
                for L in layer_ids:
                    batch_hidden = hs[L][int(b)]
                    vec = pool_number_span(tok, row_ids, batch_hidden, n_str)
                    Xb_per_layer[L].append(vec)
            for L in layer_ids:
                Xb_per_layer[L] = np.stack(Xb_per_layer[L], axis=0)
        yield Xb_per_layer, s
        if DEVICE=="cuda": torch.cuda.empty_cache()
        gc.collect()

def run_probes_for_model(model_id: str, tag: str, train_pack, eval_packs):
    print(f"\n========== {tag}: {model_id} ==========")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id if tok.eos_token_id else 0
    if tok.bos_token_id is None: tok.bos_token_id = 1
    
    model, model_type = load_dream_model(model_id)
    model.eval(); torch.set_grad_enabled(False)
    
    try:
        total_layers, width = model_layer_meta(model, model_type)
        layer_ids = list(range(1, total_layers, LAYER_STRIDE))
        print(f"Layers: {total_layers} | Width: {width}")

        scalers = {L: StandardScaler(with_mean=False) for L in layer_ids}
        texts_tr, y_tr, meta_tr, surf_tr = train_pack
        for Xb_per_layer, s in stream_hidden_features(model, model_type, tok, texts_tr, meta_tr, layer_ids, BATCH_SIZE, MAX_LEN, surf_tr):
            for L in layer_ids:
                if L in Xb_per_layer: scalers[L].partial_fit(Xb_per_layer[L])

        probes = {L: SGDClassifier(loss="log_loss", learning_rate="optimal", alpha=1e-4, tol=None, max_iter=1) for L in layer_ids}
        inited = set()
        for Xb_per_layer, s in stream_hidden_features(model, model_type, tok, texts_tr, meta_tr, layer_ids, BATCH_SIZE, MAX_LEN, surf_tr):
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
            print(f"Evaluating {name}...")
            acc = {L: [] for L in layer_ids}
            for Xb_per_layer, s in stream_hidden_features(model, model_type, tok, texts_ev, meta_ev, layer_ids, BATCH_SIZE, MAX_LEN, surf_ev):
                yb = y_ev[s]
                for L in layer_ids:
                    Xb = scalers[L].transform(Xb_per_layer[L]); pb = probes[L].predict(Xb)
                    acc[L].extend((pb == yb).tolist())
            
            acc_mean = {L: float(np.mean(acc[L])) for L in layer_ids}
            bestL = max(acc_mean, key=acc_mean.get)
            print(f"  Best layer {bestL}: acc={acc_mean[bestL]:.3f}")

            errs_by_dist = {d: [] for d in range(max_bucket_dist + 1)}
            errs_by_dir  = {-1:[], 0:[], 1:[]}
            for Xb_per_layer, s in stream_hidden_features(model, model_type, tok, texts_ev, meta_ev, [bestL], BATCH_SIZE, MAX_LEN, surf_ev):
                Xb = scalers[bestL].transform(Xb_per_layer[bestL]); yb = y_ev[s]
                proba = probes[bestL].predict_proba(Xb)[:,1]; preds = (proba >= 0.5).astype(np.int64)
                for i, idx in enumerate(range(s.start, s.stop)):
                    d, rd, n = meta_ev[idx]; d_bucket = min(d, max_bucket_dist)
                    errs_by_dist[d_bucket].append(int(preds[i] != yb[i]))
                    errs_by_dir[rd].append(int(preds[i] != yb[i]))
                        
            # Aggregate results properly
            dist_view = {k: (float(np.mean(v)) if len(v) else None, len(v)) for k,v in errs_by_dist.items()}
            dir_view  = {k: (float(np.mean(v)) if len(v) else None, len(v)) for k,v in errs_by_dir.items()}
            results[name] = {"acc_per_layer": acc_mean, "best_layer": bestL, "err_by_dist": dist_view, "err_by_dir": dir_view}
        return results
    
    finally:
        del model, tok
        if DEVICE=="cuda": torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    print("=" * 60)
    print(f"Linear Probing on Dream-v0-Instruct-7B — Target: Near-{TARGET_K}")
    print("=" * 60)
    all_results = {}
    all_results["model1"] = run_probes_for_model(MODEL_ID_1, tag="DREAM", train_pack=train_pack, eval_packs=eval_packs)
    with open(RESULT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Done. Results saved to {RESULT_FILE}")