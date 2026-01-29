import time
import importlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _truncate_on_stop(text: str, stop_sequences: Optional[List[str]]) -> str:
    if not text or not stop_sequences:
        return text
    earliest = None
    for seq in stop_sequences:
        if not seq:
            continue
        idx = text.find(seq)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    if earliest is None:
        return text
    return text[:earliest].rstrip()


def _truncate_on_patterns(text: str, stop_patterns: Optional[List[re.Pattern]]) -> str:
    if not text or not stop_patterns:
        return text
    earliest = None
    for pattern in stop_patterns:
        try:
            match = pattern.search(text)
        except Exception:
            continue
        if match:
            end = match.end()
            if earliest is None or end < earliest:
                earliest = end
    if earliest is None:
        return text
    return text[:earliest].rstrip()


def _compile_patterns(patterns: Optional[List[str]]) -> Optional[List[re.Pattern]]:
    if not patterns:
        return None
    compiled = []
    for pattern in patterns:
        if not pattern:
            continue
        try:
            compiled.append(re.compile(pattern))
        except re.error:
            continue
    return compiled or None


@dataclass
class GenerationResult:
    model: str
    provider: str
    content: str
    raw: Any
    usage: Optional[Dict[str, Any]]
    elapsed: float


class BaseProvider:
    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> GenerationResult:
        raise NotImplementedError


class OpenAIProvider(BaseProvider):
    """
    OpenAI-compatible chat completions provider.
    Works with OpenAI, vLLM, and other OpenAI-compatible servers by configuring base_url and api_key.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stop_patterns: Optional[List[str]] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.seed = seed
        self.stop = stop
        self.stop_patterns = _compile_patterns(stop_patterns)

        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "The 'openai' package is required for OpenAIProvider. Install with `pip install openai`."
            ) from e

        self._client = OpenAI(api_key=api_key, base_url=base_url) if (api_key or base_url) else OpenAI()

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> GenerationResult:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})

        start = time.time()
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            request["top_p"] = self.top_p
        if self.seed is not None:
            request["seed"] = self.seed
        if self.stop:
            request["stop"] = self.stop
        resp = self._client.chat.completions.create(**request)
        elapsed = time.time() - start

        content = resp.choices[0].message.content or ""
        content = _truncate_on_stop(content, self.stop)
        content = _truncate_on_patterns(content, self.stop_patterns)
        usage = getattr(resp, "usage", None)
        # Convert to plain dict if present
        usage_dict = None
        if usage is not None:
            try:
                usage_dict = usage.to_dict()  # type: ignore
            except Exception:
                try:
                    usage_dict = dict(usage)
                except Exception:
                    usage_dict = None
        if usage_dict is not None:
            usage_dict["token_count_source"] = "api"

        return GenerationResult(
            model=self.model,
            provider="openai-compatible",
            content=content,
            raw=resp,
            usage=usage_dict,
            elapsed=elapsed,
        )


class TransformersProvider(BaseProvider):
    """
    Local Transformers generation using `transformers` and an AutoModelForCausalLM.
    Only instantiate if you intend to run models locally.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        device_map: Optional[str] = None,
        print_device_map: bool = False,
        torch_dtype: Optional[str] = None,
        temperature: float = 0.2,
        max_new_tokens: int = 256,
        trust_remote_code: bool = False,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        diffusion_steps: Optional[int] = None,
        num_layers: Optional[int] = None,
        layer_slice_mode: str = "every_k",
        stop: Optional[List[str]] = None,
        stop_patterns: Optional[List[str]] = None,
        **config_overrides,
    ):
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoModel,
                AutoTokenizer,
                AutoConfig,
            )  # type: ignore
            import torch  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "The 'transformers' and 'torch' packages are required for TransformersProvider."
            ) from e

        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.trust_remote_code = trust_remote_code
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.diffusion_steps = diffusion_steps
        self.stop = stop
        self.stop_patterns = _compile_patterns(stop_patterns)
        self.print_device_map = print_device_map
        self.layer_slice_mode = layer_slice_mode

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=self.trust_remote_code)
        dtype = getattr(torch, torch_dtype) if torch_dtype else None

        # Load and customize config if needed
        config = None
        if num_layers is not None or config_overrides:
            try:
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=self.trust_remote_code)
                original_layers = getattr(config, 'num_hidden_layers', None)
                # Avoid shrinking config before weights load; we'll slice after load instead.
                if num_layers is not None:
                    print(
                        f"Requested num_layers={num_layers}; keeping config.num_hidden_layers={original_layers} for load."
                    )
                
                # Apply other config overrides
                for key, value in config_overrides.items():
                    if hasattr(config, key):
                        old_value = getattr(config, key)
                        setattr(config, key, value)
                        print(f"Customizing model: {key} {old_value} -> {value}")
                    else:
                        print(f"Warning: Config does not have attribute '{key}', skipping")
            except Exception as e:
                print(f"Warning: Failed to load config for customization: {e}. Loading model normally.")
                config = None

        last_err: Optional[Exception] = None
        self.model = None
        # Try common heads, then base model as a fallback for custom architectures
        load_kwargs = {
            'torch_dtype': dtype,
            'trust_remote_code': self.trust_remote_code,
        }
        if device_map:
            load_kwargs['device_map'] = device_map
            load_kwargs['low_cpu_mem_usage'] = True
        if config is not None:
            load_kwargs['config'] = config
        
        for loader_class in (AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel):
            try:
                self.model = loader_class.from_pretrained(model_name, **load_kwargs)
                break
            except Exception as e:
                last_err = e
                continue
        if self.model is None:
            raise RuntimeError(
                f"Failed to load model '{model_name}' with transformers. Last error: {last_err}"
            )
        if num_layers is not None:
            sliced = self._try_slice_layers(num_layers)
            if sliced:
                print(f"Sliced model layers to {num_layers} after load.")
            else:
                print("Warning: Unable to slice model layers after load; using loaded model as-is.")
        if device and not device_map:
            self.model.to(device)

        if self.print_device_map:
            self._print_device_allocation()

        print("Model architecture:")
        print(self.model)

        self._torch = torch

    def _print_device_allocation(self) -> None:
        try:
            import torch  # type: ignore
        except Exception:
            return

        param_counts = {}
        for name, param in self.model.named_parameters():
            try:
                dev = str(param.device)
            except Exception:
                dev = "unknown"
            param_counts[dev] = param_counts.get(dev, 0) + int(param.numel())

        if not param_counts:
            return

        total = sum(param_counts.values()) or 1
        print("Model parameter distribution by device:")
        for dev, count in sorted(param_counts.items(), key=lambda x: x[0]):
            pct = (count / total) * 100.0
            print(f"  {dev}: {count:,} params ({pct:.2f}%)")

        hf_map = getattr(self.model, "hf_device_map", None)
        if isinstance(hf_map, dict) and hf_map:
            print("Transformers device_map modules:")
            for module_name, dev in sorted(hf_map.items(), key=lambda x: x[0]):
                print(f"  {module_name}: {dev}")

    def _build_full_prompt(self, prompt: str, system: Optional[str]) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        apply = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                return (system + "\n\n" if system else "") + prompt
        return (system + "\n\n" if system else "") + prompt

    def _ensure_pad_token(self) -> None:
        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            try:
                self.tokenizer.pad_token = self.tokenizer.eos_token  # type: ignore
            except Exception:
                pass

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> GenerationResult:
        # Prefer tokenizer chat templates when available for better instruction following.
        full_prompt = self._build_full_prompt(prompt, system)

        # Ensure pad token exists for batching/generation if needed
        self._ensure_pad_token()

        start = time.time()
        content: str
        out_text: str
        usage: Optional[Dict[str, Any]] = None
        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        if str(model_type).lower() == "dream" and hasattr(self.model, "diffusion_generate"):
            out_text, content, usage = self._dream_diffusion_generate(full_prompt)
            elapsed = time.time() - start
        elif hasattr(self.model, "generate"):
            if self.seed is not None:
                self._torch.manual_seed(self.seed)
            tok = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
            stopping_criteria = None
            if self.stop or self.stop_patterns:
                try:
                    from transformers import StoppingCriteria, StoppingCriteriaList  # type: ignore

                    class StopOnSequences(StoppingCriteria):
                        def __init__(self, tokenizer, stop_sequences, stop_patterns, prompt_len):
                            self.tokenizer = tokenizer
                            self.stop_sequences = stop_sequences or []
                            self.stop_patterns = stop_patterns or []
                            self.prompt_len = prompt_len

                        def __call__(self, input_ids, scores, **kwargs):
                            gen_ids = input_ids[0][self.prompt_len :]
                            if gen_ids.numel() == 0:
                                return False
                            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                            for seq in self.stop_sequences:
                                if seq and seq in text:
                                    return True
                            if self.stop_patterns:
                                for pattern in self.stop_patterns:
                                    try:
                                        if pattern.search(text):
                                            return True
                                    except Exception:
                                        continue
                            return False

                    prompt_len = int(tok["input_ids"].shape[-1])
                    stopping_criteria = StoppingCriteriaList(
                        [StopOnSequences(self.tokenizer, self.stop, self.stop_patterns, prompt_len)]
                    )
                except Exception:
                    stopping_criteria = None

            gen_kwargs = {
                **tok,
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "stopping_criteria": stopping_criteria,
            }
            if str(model_type).lower() == "dream":
                gen_kwargs["do_sample"] = True
                if self.diffusion_steps is not None:
                    gen_kwargs["steps"] = self.diffusion_steps
            else:
                gen_kwargs["do_sample"] = self.temperature > 0
            out = self.model.generate(**gen_kwargs)
            elapsed = time.time() - start
            out_text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            # Heuristic: response is the tail beyond the prompt
            content = out_text[len(self.tokenizer.decode(tok["input_ids"][0], skip_special_tokens=True)) :].strip()
            content = _truncate_on_stop(content, self.stop)
            content = _truncate_on_patterns(content, self.stop_patterns)
            prompt_tokens = int(tok["input_ids"].shape[-1])
            completion_tokens = int(out[0].shape[-1] - tok["input_ids"].shape[-1])
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "token_count_source": "tokenizer",
            }
            if str(model_type).lower() == "dream" and self.diffusion_steps is not None:
                usage["diffusion_steps"] = self.diffusion_steps
        else:
            # Fallback for custom architectures (e.g., Dream) that expose custom generation utilities.
            out_text, content = self._custom_generate(full_prompt)
            elapsed = time.time() - start
            if self.diffusion_steps is not None:
                usage = {"diffusion_steps": self.diffusion_steps}

        return GenerationResult(
            model=self.model_name,
            provider="transformers",
            content=content,
            raw=out_text,
            usage=usage,
            elapsed=elapsed,
        )

    def _dream_diffusion_generate(self, full_prompt: str) -> tuple[str, str, Optional[Dict[str, Any]]]:
        tok = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        prompt_len = int(tok["input_ids"].shape[-1])
        mask_token_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            mask_token_id = getattr(getattr(self.model, "config", None), "mask_token_id", None)
        gen_kwargs = {
            "inputs": tok["input_ids"],
            "attention_mask": tok.get("attention_mask"),
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        if self.diffusion_steps is not None:
            gen_kwargs["steps"] = self.diffusion_steps
        if mask_token_id is not None:
            gen_kwargs["mask_token_id"] = mask_token_id
        out = self.model.diffusion_generate(**gen_kwargs)
        sequences = getattr(out, "sequences", out)
        out_ids = sequences[0]
        out_text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        gen_ids = out_ids[prompt_len:]
        content = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        content = _truncate_on_stop(content, self.stop)
        content = _truncate_on_patterns(content, self.stop_patterns)
        completion_tokens = int(out_ids.shape[-1] - prompt_len)
        usage = {
            "prompt_tokens": prompt_len,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_len + completion_tokens,
            "token_count_source": "tokenizer",
        }
        if self.diffusion_steps is not None:
            usage["diffusion_steps"] = self.diffusion_steps
        return out_text, content, usage

    def layerwise_metrics(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 256,
        token_strategy: str = "last",
        seed: Optional[int] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        import numpy as np  # type: ignore

        full_prompt = self._build_full_prompt(prompt, system)
        self._ensure_pad_token()
        tok = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
        ).to(self.model.device)
        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        if str(model_type).lower() == "dream" and "attention_mask" in tok:
            tok["attention_mask"] = tok["attention_mask"].to(dtype=self._torch.bool)

        with self._torch.no_grad():
            outputs = self.model(**tok, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states

        if hidden_states is None:
            return {}

        if seed is not None:
            self._torch.manual_seed(seed)

        num_layers = len(hidden_states)
        metric_set = {m.strip() for m in metrics} if metrics else None
        spectral_entropies: List[float] = []
        effective_ranks: List[int] = []
        activation_entropies: List[float] = []
        trace_covariances: List[float] = []
        gradient_norms: List[float] = []
        cosine_similarities: List[float] = []
        l2_distances: List[float] = []
        angular_distances: List[float] = []

        layer_means: List[np.ndarray] = []

        for layer_idx, hidden in enumerate(hidden_states):
            if hidden is None:
                if metric_set is None or "spectral_entropies" in metric_set:
                    spectral_entropies.append(0.0)
                if metric_set is None or "effective_ranks" in metric_set:
                    effective_ranks.append(0)
                if metric_set is None or "activation_entropies" in metric_set:
                    activation_entropies.append(0.0)
                if metric_set is None or "trace_covariances" in metric_set:
                    trace_covariances.append(0.0)
                if metric_set is None or "gradient_norms" in metric_set:
                    gradient_norms.append(0.0)
                layer_means.append(None)  # type: ignore[list-item]
                continue

            hidden_cpu = hidden.detach().float().cpu()
            seq_len = hidden_cpu.shape[1]
            take = min(max_tokens, seq_len)
            if take <= 0:
                if metric_set is None or "spectral_entropies" in metric_set:
                    spectral_entropies.append(0.0)
                if metric_set is None or "effective_ranks" in metric_set:
                    effective_ranks.append(0)
                if metric_set is None or "activation_entropies" in metric_set:
                    activation_entropies.append(0.0)
                if metric_set is None or "trace_covariances" in metric_set:
                    trace_covariances.append(0.0)
                if metric_set is None or "gradient_norms" in metric_set:
                    gradient_norms.append(0.0)
                layer_means.append(None)  # type: ignore[list-item]
                continue

            if token_strategy == "first":
                sample = hidden_cpu[:, :take, :]
            elif token_strategy == "random":
                idx = self._torch.randperm(seq_len)[:take]
                sample = hidden_cpu.index_select(1, idx)
            else:
                sample = hidden_cpu[:, -take:, :]

            flat = sample.reshape(-1, sample.shape[-1]).numpy()
            layer_mean = flat.mean(axis=0) if flat.size else None
            layer_means.append(layer_mean)

            # Spectral metrics
            if metric_set is None or "spectral_entropies" in metric_set or "effective_ranks" in metric_set:
                try:
                    s = np.linalg.svd(flat, compute_uv=False)
                    if s.size == 0:
                        if metric_set is None or "spectral_entropies" in metric_set:
                            spectral_entropies.append(0.0)
                        if metric_set is None or "effective_ranks" in metric_set:
                            effective_ranks.append(0)
                    else:
                        s_norm = s / (s.sum() + 1e-12)
                        if metric_set is None or "spectral_entropies" in metric_set:
                            spectral_entropies.append(float(-(s_norm * np.log(s_norm + 1e-12)).sum()))
                        if metric_set is None or "effective_ranks" in metric_set:
                            threshold = 0.01 * float(s.max())
                            effective_ranks.append(int((s > threshold).sum()))
                except Exception:
                    if metric_set is None or "spectral_entropies" in metric_set:
                        spectral_entropies.append(0.0)
                    if metric_set is None or "effective_ranks" in metric_set:
                        effective_ranks.append(0)

            # Activation entropy
            if metric_set is None or "activation_entropies" in metric_set:
                try:
                    hist, _ = np.histogram(flat, bins=50, density=True)
                    hist = hist[hist > 0]
                    activation_entropies.append(float(-(hist * np.log(hist)).sum()))
                except Exception:
                    activation_entropies.append(0.0)

            # Trace of covariance
            if metric_set is None or "trace_covariances" in metric_set:
                try:
                    var = flat.var(axis=0)
                    trace_covariances.append(float(var.sum()))
                except Exception:
                    trace_covariances.append(0.0)

            # Gradient norm proxy (variance-based)
            if metric_set is None or "gradient_norms" in metric_set:
                try:
                    variance = float(flat.var())
                    gradient_norms.append(float((variance ** 0.5) * flat.size))
                except Exception:
                    gradient_norms.append(0.0)

        if metric_set is None or any(
            m in metric_set for m in ("cosine_similarities", "l2_distances", "angular_distances")
        ):
            for idx in range(num_layers - 1):
                a = layer_means[idx]
                b = layer_means[idx + 1]
                if a is None or b is None:
                    if metric_set is None or "cosine_similarities" in metric_set:
                        cosine_similarities.append(0.0)
                    if metric_set is None or "l2_distances" in metric_set:
                        l2_distances.append(0.0)
                    if metric_set is None or "angular_distances" in metric_set:
                        angular_distances.append(0.0)
                    continue
                denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
                cos = float(np.dot(a, b) / denom)
                if metric_set is None or "cosine_similarities" in metric_set:
                    cosine_similarities.append(cos)
                if metric_set is None or "l2_distances" in metric_set:
                    l2_distances.append(float(np.linalg.norm(a - b)))
                if metric_set is None or "angular_distances" in metric_set:
                    cos_clamped = min(1.0, max(-1.0, cos))
                    angular_distances.append(float(np.arccos(cos_clamped)))

        payload: Dict[str, Any] = {"num_layers": num_layers, "token_strategy": token_strategy, "max_tokens": max_tokens}
        if metric_set is None or "spectral_entropies" in metric_set:
            payload["spectral_entropies"] = spectral_entropies
        if metric_set is None or "effective_ranks" in metric_set:
            payload["effective_ranks"] = effective_ranks
        if metric_set is None or "activation_entropies" in metric_set:
            payload["activation_entropies"] = activation_entropies
        if metric_set is None or "trace_covariances" in metric_set:
            payload["trace_covariances"] = trace_covariances
        if metric_set is None or "gradient_norms" in metric_set:
            payload["gradient_norms"] = gradient_norms
        if metric_set is None or "cosine_similarities" in metric_set:
            payload["cosine_similarities"] = cosine_similarities
        if metric_set is None or "l2_distances" in metric_set:
            payload["l2_distances"] = l2_distances
        if metric_set is None or "angular_distances" in metric_set:
            payload["angular_distances"] = angular_distances
        return payload

    def _try_slice_layers(self, num_layers: int) -> bool:
        """Best-effort slice of transformer layers after loading pretrained weights."""
        candidates = []
        for path in ("model.layers", "transformer.h", "encoder.layers", "decoder.layers"):
            obj = self.model
            ok = True
            for attr in path.split("."):
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok:
                candidates.append((path, obj))

        module_list_type = None
        if getattr(self, "_torch", None) is not None:
            module_list_type = getattr(getattr(self._torch, "nn", object), "ModuleList", None)
        else:
            try:
                import torch  # type: ignore

                module_list_type = getattr(getattr(torch, "nn", object), "ModuleList", None)
            except Exception:
                module_list_type = None

        def _select_indices(total: int, keep: int, mode: str) -> list[int]:
            if keep <= 0:
                return []
            if keep == 1:
                return [total - 1]
            if mode == "last_n":
                start = max(0, total - keep)
                indices = list(range(start, total))
                if indices and indices[0] != 0:
                    indices[0] = 0
                return sorted(set(indices))[:keep]
            if mode == "first_n":
                indices = list(range(0, min(keep, total)))
                if indices and indices[-1] != total - 1:
                    indices[-1] = total - 1
                return sorted(set(indices))[:keep]
            # every_k (default)
            if total % keep == 0:
                stride = total // keep
                indices = list(range(0, total, stride))
                if indices and indices[-1] != total - 1:
                    indices = indices[:-1] + [total - 1]
            else:
                step = (total - 1) / (keep - 1)
                indices = [int(round(i * step)) for i in range(keep)]
            return indices[:keep]

        for path, layers in candidates:
            if module_list_type is not None:
                is_sequence = isinstance(layers, (list, tuple, module_list_type))
            else:
                is_sequence = isinstance(layers, (list, tuple))
            if not is_sequence:
                continue
            total_layers = len(layers)
            if total_layers <= num_layers:
                return True
            if num_layers <= 0:
                return False
            try:
                layer_list = list(layers)
                mode = self.layer_slice_mode or "every_k"
                indices = _select_indices(total_layers, num_layers, mode)
                print(f"Sliced layer indices ({mode}): {indices}")
                kept = [layer_list[i] for i in indices]
                parent = self.model
                parts = path.split(".")
                for attr in parts[:-1]:
                    parent = getattr(parent, attr)
                if module_list_type is not None and isinstance(layers, module_list_type):
                    new_layers = module_list_type(kept)
                else:
                    new_layers = type(layers)(kept)
                setattr(parent, parts[-1], new_layers)
                if hasattr(self.model, "config"):
                    setattr(self.model.config, "num_hidden_layers", num_layers)
                return True
            except Exception:
                continue
        return False

    def _custom_generate(self, full_prompt: str) -> tuple[str, str]:
        """Attempt custom text generation when model has no .generate().

        Strategy:
        - Look up the config module to infer the local transformers_modules package path.
        - Import a sibling module named 'generation_utils' and try calling 'diffusion_generate'.
        - Try a few common call signatures and extract a string result.
        """
        # Compose module path based on config module
        cfg_mod = getattr(getattr(self.model, "config", object), "__class__", object).__module__
        gen_mod_name = None
        if isinstance(cfg_mod, str) and ".configuration_" in cfg_mod:
            gen_mod_name = cfg_mod.rsplit(".", 1)[0].replace("configuration_", "generation_")
        # Fallback: try replacing the last component with 'generation_utils'
        if not gen_mod_name and isinstance(cfg_mod, str):
            parts = cfg_mod.split(".")
            if len(parts) >= 1:
                parts[-1] = "generation_utils"
                gen_mod_name = ".".join(parts)

        func = None
        gen_mod = None
        if gen_mod_name:
            try:
                gen_mod = importlib.import_module(gen_mod_name)
                func = getattr(gen_mod, "diffusion_generate", None)
            except Exception:
                func = None

        if callable(func):
            # Try a few signatures
            for call in (
                lambda: func(
                    self.model,
                    self.tokenizer,
                    [full_prompt],
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    **({"num_steps": self.diffusion_steps} if self.diffusion_steps is not None else {}),
                ),
                lambda: func(
                    self.model,
                    self.tokenizer,
                    [full_prompt],
                    max_new_tokens=self.max_new_tokens,
                    **({"num_steps": self.diffusion_steps} if self.diffusion_steps is not None else {}),
                ),
                lambda: func(self.model, self.tokenizer, [full_prompt]),
            ):
                try:
                    result = call()
                    # Normalize result to string
                    if isinstance(result, str):
                        return result, result
                    if isinstance(result, (list, tuple)) and result and isinstance(result[0], str):
                        return result[0], result[0]
                    if isinstance(result, dict):
                        # Common keys
                        for k in ("text", "content", "output", "generated_text"):
                            if isinstance(result.get(k), str):
                                s = result[k]
                                return s, s
                    # Fallback string cast
                    return str(result), str(result)
                except Exception:
                    continue

        # Ultimate fallback: echo the prompt to avoid crashes
        return full_prompt, full_prompt


class OllamaProvider(BaseProvider):
    """Local Ollama provider using the `ollama` Python package."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        num_predict: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stop_patterns: Optional[List[str]] = None,
    ):
        try:
            import ollama  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("The 'ollama' package is required for OllamaProvider.") from e

        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.stop = stop
        self.stop_patterns = _compile_patterns(stop_patterns)
        self._ollama = ollama

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> GenerationResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        options: Dict[str, Any] = {"temperature": self.temperature}
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.top_k is not None:
            options["top_k"] = self.top_k
        if self.seed is not None:
            options["seed"] = self.seed
        if self.stop:
            options["stop"] = self.stop
        resp = self._ollama.chat(
            model=self.model,
            messages=messages,
            options=options,
        )
        elapsed = time.time() - start
        content = resp.get("message", {}).get("content", "")
        content = _truncate_on_stop(content, self.stop)
        content = _truncate_on_patterns(content, self.stop_patterns)

        return GenerationResult(
            model=self.model,
            provider="ollama",
            content=content,
            raw=resp,
            usage=None,
            elapsed=elapsed,
        )
