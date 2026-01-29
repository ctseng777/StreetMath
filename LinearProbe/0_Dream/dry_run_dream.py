#!/usr/bin/env python3
"""
[DRY RUN] Linear probe script for Dream-v0-Instruct-7B diffusion language model.
This is a patched version of linear_probe_dream_fixed.py for quick testing.
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

# >>>>> DRY RUN CONFIG CHANGE <<<<<
TRAIN_N = 20   # Changed from 2000
VAL_N   = 10   # Changed from 500
BATCH_SIZE = 2 # Small batch size
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

MAX_LEN     = 48     
LAYER_STRIDE = 1     

NEAR_THRESHOLD = 1

# -----------------------
# Data generation (same as original)
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

def dist_to_nearest_5(n: int) -> int:
    last = n % 10
    return min(abs(last-0), abs(last-5), abs(last-10))

def rounding_direction(n: int) -> int:
    d0 = abs((n//5)*5 - n)
    d1 = abs(((n+4)//5)*5 - n)
    if d0 == 0 or d1 == 0:
        return 0
    return +1 if d1 < d0 else -1

def make_dataset(N: int, templates: List[str], surface: str = "digits", lo: int = 0, hi: int = 9999, seed: int = 1337):
    rng = random.Random(seed)
    texts, labels, metas, spans = [], [], [], []
    for _ in range(N):
        n = rng.randint(lo, hi)
        d = dist_to_nearest_5(n)
        y = 1 if d <= NEAR_THRESHOLD else 0
        rd = rounding_direction(n)
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

# Generate datasets
print(f"Generating DRY RUN data: Train={TRAIN_N}, Val={VAL_N}")
train_texts, y_train, meta_train = make_dataset(TRAIN_N, TEMPLATES_A, surface="digits", seed=SEED)
valA_texts,  y_valA,  meta_valA  = make_dataset(VAL_N,   TEMPLATES_B, surface="digits", seed=SEED+1)
valW_texts,  y_valW,  meta_valW  = make_dataset(VAL_N,   TEMPLATES_A, surface="words",  seed=SEED+2)

# -----------------------
# Token span finder (same as original)
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
        # Ensure we're indexing with CPU tensor or int
        idx = T - 1
        return layer_h[idx].detach().cpu().numpy()
    s, e = span
    # Ensure indices are integers, not tensors
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
        # Load config first to check architecture
        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        # Load the Dream model for causal LM
        # Don't use device_map="auto" to avoid split across devices
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Move entire model to CUDA if available
        if DEVICE == "cuda":
            model = model.to(DEVICE)
            
        print(f"  Successfully loaded Dream model")
        print(f"  Model type: {config.model_type if hasattr(config, 'model_type') else 'dream'}")
        print(f"  Architecture: {config.architectures if hasattr(config, 'architectures') else 'DreamForCausalLM'}")
        
        return model, "dream"
        
    except Exception as e:
        print(f"  Warning: Error loading with AutoModelForCausalLM: {e}")
        print("  Attempting fallback to AutoModel...")
        
        # Fallback to AutoModel
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
            trust_remote_code=True
        )
        
        if DEVICE == "cuda":
            model = model.to(DEVICE)
            
        print("  Loaded with AutoModel fallback")
        return model, "auto"

# -----------------------
# Streaming helpers (modified for Dream)
# -----------------------
def iter_batches(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs], slice(i, min(i+bs, len(lst)))

def model_layer_meta(model, model_type: str) -> Tuple[int,int]:
    """Get layer count and hidden dimension from Dream model."""
    with torch.no_grad():
        # Get model name for tokenizer
        if hasattr(model, 'name_or_path'):
            tok_path = model.name_or_path
        elif hasattr(model, 'config') and hasattr(model.config, '_name_or_path'):
            tok_path = model.config._name_or_path
        else:
            tok_path = MODEL_ID_1
            
        # Initialize tokenizer
        tmp_tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        if tmp_tok.pad_token_id is None: 
            tmp_tok.pad_token_id = tmp_tok.eos_token_id if tmp_tok.eos_token_id is not None else 0
            
        # Prepare test input
        test_text = "The number 42"
        enc = tmp_tok(test_text, return_tensors="pt", padding=False, truncation=True, max_length=16)
        
        # Move to device
        inputs = {}
        for k, v in enc.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(DEVICE)
        
        # Get outputs with hidden states
        try:
            if model_type == "dream" and hasattr(model, 'model'):
                # For Dream CausalLM wrapper, access the base model
                out = model(**inputs, use_cache=False, output_hidden_states=True, return_dict=True)
            else:
                out = model(**inputs, use_cache=False, output_hidden_states=True, return_dict=True)
                
            # Extract layer count and dimension
            if hasattr(out, 'hidden_states') and out.hidden_states:
                L = len(out.hidden_states)
                D = out.hidden_states[-1].shape[-1]
            else:
                # Fallback: try to get from config
                config = model.config
                L = getattr(config, 'num_hidden_layers', 28) + 1  # +1 for embeddings
                D = getattr(config, 'hidden_size', 3584)
                print(f"  Warning: Using config values L={L}, D={D}")
                
        except Exception as e:
            print(f"  Warning: Error getting layer meta: {e}")
            # Use correct values for Dream-7B from config.json
            L = 29  # 28 hidden layers + 1 embedding layer
            D = 3584  # actual hidden size from config
            print(f"  Using Dream-7B config values: L={L}, D={D}")
                
        del enc, inputs
        if 'out' in locals():
            del out
        if DEVICE == "cuda": 
            torch.cuda.empty_cache()
        gc.collect()
        
        return L, D

def stream_hidden_features(model, model_type, tok, texts, metas, layer_ids, batch_size, max_len, surface):
    """Stream hidden features from Dream model for each batch."""
    for chunk, s in iter_batches(texts, batch_size):
        # Tokenize batch
        enc = tok(chunk, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
        
        # Move to device
        input_ids = enc["input_ids"].to(DEVICE)
        
        # Skip empty batches
        if input_ids.size(0) == 0:
            continue
            
        with torch.no_grad():
            try:
                # For Dream model, run without attention mask
                # The model handles padding internally
                outputs = model(
                    input_ids=input_ids,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Extract hidden states
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    hs = outputs.hidden_states
                else:
                    print(f"  Warning: No hidden states in output, skipping batch")
                    continue
                    
            except Exception as e:
                print(f"  Warning: Skipping batch due to error: {e}")
                continue
                
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
                    # Ensure hidden state tensor is on correct device and use integer indexing
                    layer_hidden = hs[L]
                    if layer_hidden.device != torch.device(DEVICE):
                        layer_hidden = layer_hidden.to(DEVICE)
                    # Use integer indexing to avoid device mismatch
                    batch_hidden = layer_hidden[int(b)]
                    vec = pool_number_span(tok, row_ids, batch_hidden, n_str)
                    Xb_per_layer[L].append(vec)
                        
            for L in layer_ids:
                Xb_per_layer[L] = np.stack(Xb_per_layer[L], axis=0)
                    
        yield Xb_per_layer, s
        del outputs, hs, enc, input_ids, Xb_per_layer
        if DEVICE == "cuda": torch.cuda.empty_cache()
        gc.collect()

# -----------------------
# Main probing pipeline
# -----------------------
def run_probes_for_model(model_id: str, tag: str, train_pack, eval_packs):
    print(f"\n========== {tag}: {model_id} ==========")
    
    # Load tokenizer with Dream-specific settings
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None: 
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    
    # Add special tokens if needed
    if tok.bos_token_id is None:
        tok.bos_token_id = 1
    
    # Load Dream model
    model, model_type = load_dream_model(model_id)
    model.eval()
    torch.set_grad_enabled(False)
    
    # Get model metadata
    total_layers, width = model_layer_meta(model, model_type)
        
    layer_ids = list(range(1, total_layers, LAYER_STRIDE))  # Skip embeddings
    print(f"Layers: {total_layers} | width: {width} | probing layers: {layer_ids[:8]}{'...' if len(layer_ids)>8 else ''}")
    
    # Check model architecture details
    if hasattr(model, 'config'):
        config = model.config
        print(f"  Model config: layers={getattr(config, 'num_hidden_layers', 'unknown')}, "
              f"hidden_size={getattr(config, 'hidden_size', 'unknown')}, "
              f"vocab_size={getattr(config, 'vocab_size', 'unknown')}")
    
    # Note about Dream model
    print("  Note: Dream model uses diffusion-based generation but standard forward pass for hidden states")
    
    # First pass: fit StandardScalers
    scalers = {L: StandardScaler(with_mean=False) for L in layer_ids}
    texts_tr, y_tr, meta_tr, surf_tr = train_pack
    
    print("  Pass 1: Fitting scalers...")
    for Xb_per_layer, s in stream_hidden_features(model, model_type, tok, texts_tr, meta_tr, layer_ids, BATCH_SIZE, MAX_LEN, surf_tr):
        for L in layer_ids:
            if L in Xb_per_layer:
                scalers[L].partial_fit(Xb_per_layer[L])
    
    # Second pass: fit SGD classifiers
    probes = {L: SGDClassifier(loss="log_loss", learning_rate="optimal", alpha=1e-4, tol=None, max_iter=1) for L in layer_ids}
    inited = set()
    
    print("  Pass 2: Training probes...")
    for Xb_per_layer, s in stream_hidden_features(model, model_type, tok, texts_tr, meta_tr, layer_ids, BATCH_SIZE, MAX_LEN, surf_tr):
        yb = y_tr[s]
        for L in layer_ids:
            Xb = scalers[L].transform(Xb_per_layer[L])
            if L not in inited:
                probes[L].partial_fit(Xb, yb, classes=np.array([0,1], dtype=np.int64))
                inited.add(L)
            else:
                probes[L].partial_fit(Xb, yb)
    
    # Evaluation
    results = {}
    for name, (texts_ev, y_ev, meta_ev, surf_ev) in eval_packs.items():
        print(f"  Evaluating {name}...")
        acc = {L: [] for L in layer_ids}
        
        for Xb_per_layer, s in stream_hidden_features(model, model_type, tok, texts_ev, meta_ev, layer_ids, BATCH_SIZE, MAX_LEN, surf_ev):
            yb = y_ev[s]
            for L in layer_ids:
                Xb = scalers[L].transform(Xb_per_layer[L])
                pb = probes[L].predict(Xb)
                acc[L].extend((pb == yb).tolist())
        
        # Calculate accuracies
        acc_mean = {L: float(np.mean(acc[L])) for L in layer_ids}
        bestL = max(acc_mean, key=acc_mean.get)
        print(f"    Best layer {bestL}: acc={acc_mean[bestL]:.3f}")
        
        # Error breakdown for best layer
        errs_by_dist = {0:[], 1:[], 2:[]}
        errs_by_dir  = {-1:[], 0:[], +1:[]}
        
        for Xb_per_layer, s in stream_hidden_features(model, model_type, tok, texts_ev, meta_ev, [bestL], BATCH_SIZE, MAX_LEN, surf_ev):
            Xb = scalers[bestL].transform(Xb_per_layer[bestL])
            yb = y_ev[s]
            proba = probes[bestL].predict_proba(Xb)[:,1]
            preds = (proba >= 0.5).astype(np.int64)
            for i, idx in enumerate(range(s.start, s.stop)):
                d, rd, n = meta_ev[idx]
                d_bucket = min(d, 2)
                errs_by_dist[d_bucket].append(int(preds[i] != yb[i]))
                errs_by_dir[rd].append(int(preds[i] != yb[i]))
                    
        dist_view = {k: (float(np.mean(v)) if len(v) else None, len(v)) for k,v in errs_by_dist.items()}
        dir_view  = {k: (float(np.mean(v)) if len(v) else None, len(v)) for k,v in errs_by_dir.items()}
        results[name] = {"acc_per_layer": acc_mean, "best_layer": bestL, "err_by_dist": dist_view, "err_by_dir": dir_view}
    
    # Clean up
    del model, tok
    if DEVICE=="cuda": torch.cuda.empty_cache()
    gc.collect()
    return results

# -----------------------
# Run experiments
# -----------------------
train_pack = (train_texts, y_train, meta_train, "digits")
eval_packs = {
    "digits_paraphrase": (valA_texts, y_valA, meta_valA, "digits"),
    "words":             (valW_texts, y_valW, meta_valW, "words"),
}

print("=" * 60)
print("Linear Probing on Dream-v0-Instruct-7B Diffusion LM [DRY RUN MODE]")
print(f"Device: {DEVICE} | Dtype: {DTYPE}")
print("=" * 60)

all_results = {}
# 执行一次跑通
all_results["model1"] = run_probes_for_model(MODEL_ID_1, tag="DREAM", train_pack=train_pack, eval_packs=eval_packs)

# Print summary
if all_results:
    print("\n================ SUMMARY ================")
    for mid, res in all_results.items():
        print(f"\n--- {mid} ---")
        for split, info in res.items():
            if info:
                Lbest = info["best_layer"]
                acc = info["acc_per_layer"][Lbest]
                print(f"{split:18s} -> best layer {Lbest:2d} acc={acc:.3f}")
else:
    print("\n⚠ No results generated.")