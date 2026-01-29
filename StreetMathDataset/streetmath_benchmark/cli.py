import argparse
import glob
import json
import os
from typing import Optional

from .loader import load_streetmath, resolve_device_map
from .prompt import build_prompt, DEFAULT_SYSTEM_PROMPT
from .providers import OpenAIProvider, TransformersProvider, OllamaProvider
from .eval import build_result_record, summarize


def make_provider(args):
    provider = args.provider
    if provider == "openai":
        return OpenAIProvider(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            seed=args.seed,
            stop=args.stop,
            stop_patterns=args.stop_pattern,
        )
    elif provider == "transformers":
        return TransformersProvider(
            model_name=args.model,
            device=args.device,
            device_map=args.device_map,
            print_device_map=args.print_device_map,
            torch_dtype=args.torch_dtype,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens or 256,
            trust_remote_code=args.trust_remote_code,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            diffusion_steps=args.diffusion_steps,
            num_layers=getattr(args, 'num_layers', None),
            layer_slice_mode=args.layer_slice_mode,
            stop=args.stop,
            stop_patterns=args.stop_pattern,
        )
    elif provider == "ollama":
        return OllamaProvider(
            model=args.model,
            temperature=args.temperature,
            num_predict=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            stop=args.stop,
            stop_patterns=args.stop_pattern,
        )
    else:
        raise SystemExit(f"Unknown provider: {provider}")


def run():
    parser = argparse.ArgumentParser(description="StreetMath benchmark runner")
    parser.add_argument("--provider", choices=["openai", "transformers", "ollama"], default="openai")
    parser.add_argument("--model", required=True, help="Model name/ID (e.g., qwen/Qwen2-7B-Instruct)")
    parser.add_argument("--api-key", dest="api_key", default=os.getenv("RUNPOD_API_KEY"))
    parser.add_argument("--base-url", dest="base_url", default=os.getenv("RUNPOD_BASE_URL"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None, help="Max new tokens (or completion tokens)")
    parser.add_argument(
        "--stop",
        action="append",
        default=None,
        help="Stop sequence(s). Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--stop-pattern",
        action="append",
        default=None,
        help="Stop regex pattern(s). Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--stop-at-final-answer",
        action="store_true",
        help="Stop once a 'Final answer: <number>' line appears.",
    )

    # Transformers-only options
    parser.add_argument("--device", default=None, help="Device for transformers (e.g., cuda:0)")
    parser.add_argument(
        "--device-map",
        default=None,
        help="Transformers device_map (e.g., auto, balanced). Enables multi-GPU sharding when set.",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Convenience flag to set device_map=auto when multiple CUDA devices are available.",
    )
    parser.add_argument(
        "--print-device-map",
        action="store_true",
        help="Print model parameter distribution across devices (transformers only).",
    )
    parser.add_argument("--torch-dtype", default=None, help="Torch dtype (e.g., float16)")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code on loading")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of transformer layers to keep when slicing (preserves first/last when possible).",
    )
    parser.add_argument(
        "--layer-slice-mode",
        choices=["every_k", "last_n", "first_n"],
        default="every_k",
        help="Layer slicing strategy when --num-layers is set.",
    )

    # Dataset options
    parser.add_argument("--dataset", default="LuxMuseAI/StreetMathDataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--local-jsonl", dest="local_jsonl", default=None, help="Path to local JSONL copy")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--samples-percentage",
        type=int,
        default=100,
        help="Percentage of samples to take per topic (1-100).",
    )

    # Prompt options
    parser.add_argument("--no-system", action="store_true", help="Do not send a system prompt")
    parser.add_argument("--custom-system", default=None, help="Custom system prompt text")
    parser.add_argument("--no-tools", action="store_true", help="Disallow tool or function calls in responses")
    parser.add_argument("--hint", action="store_true", help="Add hint block to free-form instructions.")
    parser.add_argument(
        "--prompt-template-file",
        default=None,
        help="Path to a file containing custom user instructions appended to each sample.",
    )

    # Output
    parser.add_argument("--output", required=True, help="Path to benchmark output JSONL file")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output file.")
    parser.add_argument("--resume-errors", action="store_true", help="Re-run samples that previously errored.")
    parser.add_argument("--log-file", default=None, help="Optional log file for errors/progress.")
    parser.add_argument("--log-raw-response", action="store_true", help="Include raw model response payload in records.")
    parser.add_argument(
        "--arch",
        choices=["dense", "moe", "diffusion", "ssm"],
        default=None,
        help="Architecture for FLOPs estimation.",
    )
    parser.add_argument("--model-params", type=float, default=None, help="Model parameter count (e.g., 4e9).")
    parser.add_argument("--active-params", type=float, default=None, help="Active params for MoE models.")
    parser.add_argument("--diffusion-steps", type=int, default=None, help="Diffusion steps for diffusion models.")
    parser.add_argument("--layerwise", action="store_true", help="Collect layerwise metrics per prompt (transformers only).")
    parser.add_argument("--layerwise-output", default=None, help="Path to layerwise JSONL output (files will be split every 100 samples with sequence numbers).")
    parser.add_argument("--layerwise-summary-only", action="store_true", help="Store only aggregated layerwise metrics.")
    parser.add_argument("--layerwise-embed", action="store_true", help="Embed layerwise metrics into main output records.")
    parser.add_argument("--layerwise-max-tokens", type=int, default=256, help="Max tokens for layerwise analysis.")
    parser.add_argument(
        "--layerwise-token-strategy",
        choices=["last", "first", "random"],
        default="last",
        help="Token selection strategy for layerwise metrics.",
    )
    parser.add_argument("--layerwise-seed", type=int, default=None, help="Seed for layerwise token sampling.")
    parser.add_argument(
        "--layerwise-metrics",
        default=None,
        help="Comma-separated list of layerwise metrics to collect (e.g., spectral_entropies,effective_ranks).",
    )

    args = parser.parse_args()

    def parse_stop(values):
        if not values:
            return None
        stops = []
        for value in values:
            if not value:
                continue
            for part in value.split(","):
                part = part.strip()
                if part:
                    stops.append(part)
        return stops or None

    args.stop = parse_stop(args.stop)
    args.stop_pattern = parse_stop(args.stop_pattern)
    if args.stop_at_final_answer:
        final_pattern = r"(?i)Final answer:\s*[-+]?\d+(?:\.\d+)?\s*\n"
        if args.stop_pattern:
            args.stop_pattern.append(final_pattern)
        else:
            args.stop_pattern = [final_pattern]

    if args.device and (args.device_map or args.multi_gpu):
        print("Warning: --device set; ignoring --device-map/--multi-gpu.")
        args.device_map = None
        args.multi_gpu = False
    args.device_map = resolve_device_map(args.device_map, args.multi_gpu)

    try:
        provider = make_provider(args)
    except Exception as e:
        raise SystemExit(f"Failed to initialize provider: {e}")

    print(f"Loading dataset: {args.dataset}:{args.split} (limit={args.limit}, shuffle={args.shuffle})")
    if args.samples_percentage < 1 or args.samples_percentage > 100:
        raise SystemExit("--samples-percentage must be between 1 and 100.")

    samples = load_streetmath(
        dataset=args.dataset,
        split=args.split,
        shuffle=args.shuffle,
        seed=args.seed,
        limit=args.limit,
        local_jsonl=args.local_jsonl,
    )
    if args.samples_percentage < 100:
        by_topic = {}
        for s in samples:
            topic = s.get("topic") or "unknown"
            by_topic.setdefault(topic, []).append(s)
        reduced = []
        for topic, items in by_topic.items():
            keep = max(1, int(len(items) * args.samples_percentage / 100))
            reduced.extend(items[:keep])
        samples = reduced
    print(f"Loaded {len(samples)} samples.")

    system_prompt: Optional[str] = None
    if not args.no_system:
        system_prompt = args.custom_system or DEFAULT_SYSTEM_PROMPT

    # Prepare output file
    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    results = []
    overall_total = len(samples)
    print(f"Starting benchmark on model {args.model} via {args.provider}...")

    custom_user = None
    if args.prompt_template_file:
        try:
            with open(args.prompt_template_file, "r", encoding="utf-8") as f:
                custom_user = f.read()
        except Exception as e:
            raise SystemExit(f"Failed to read --prompt-template-file: {e}")

    def log_line(message: str) -> None:
        if not args.log_file:
            return
        with open(args.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    processed_success = set()
    processed_error = set()
    resume_offset = 0
    if args.resume and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if "summary" in obj:
                    continue
                rec_id = obj.get("id")
                if not rec_id:
                    continue
                if "judgement" in obj:
                    processed_success.add(rec_id)
                    results.append(obj)
                elif "error" in obj:
                    processed_error.add(rec_id)
                    if not args.resume_errors:
                        results.append(obj)
        print(f"Resuming: {len(processed_success)} completed, {len(processed_error)} errored.")
        log_line(f"Resuming: {len(processed_success)} completed, {len(processed_error)} errored.")
        # Fast-forward over a contiguous processed prefix to avoid re-iterating from the start.
        for idx, sample in enumerate(samples, 1):
            sample_id = sample.get("id")
            if not sample_id:
                break
            if sample_id in processed_success:
                resume_offset = idx
                continue
            if sample_id in processed_error and not args.resume_errors:
                resume_offset = idx
                continue
            break
        if resume_offset:
            print(f"Resuming: skipping first {resume_offset} samples already recorded.")
            log_line(f"Resuming: skipping first {resume_offset} samples already recorded.")
            samples = samples[resume_offset:]

    layerwise_writer_state = None
    layerwise_summary = None
    layerwise_processed_ids = set()  # Track IDs already written to layerwise files
    if args.layerwise:
        if args.provider != "transformers":
            print("Layerwise metrics only supported for --provider transformers; ignoring --layerwise.")
        else:
            if not args.layerwise_output and not args.layerwise_embed:
                raise SystemExit("Layerwise enabled but no output requested. Use --layerwise-output or --layerwise-embed.")
            if args.layerwise_summary_only and args.layerwise_embed:
                print("--layerwise-summary-only set; ignoring --layerwise-embed to avoid per-sample payloads.")
            if args.layerwise_output:
                # Parse base filename to generate sequence-numbered files
                base_path = args.layerwise_output
                # Remove .gz extension if present (for backward compatibility)
                if base_path.endswith(".gz"):
                    base_path = base_path[:-3]
                # Split into directory, base name, and extension
                dir_name = os.path.dirname(base_path) or "."
                base_name = os.path.basename(base_path)
                if "." in base_name:
                    name_part, ext = base_name.rsplit(".", 1)
                    ext = "." + ext
                else:
                    name_part = base_name
                    ext = ""
                # If a user provided an already-sequenced filename, normalize it back to the base.
                if len(name_part) > 4 and name_part[-4] == "_" and name_part[-3:].isdigit():
                    name_part = name_part[:-4]
                
                def get_layerwise_filename(seq_num: int) -> str:
                    return os.path.join(dir_name, f"{name_part}_{seq_num:03d}{ext}")
                
                # Scan existing layerwise files for resume
                current_seq = 0
                samples_in_file = 0
                if args.resume:
                    # Find all existing layerwise files
                    pattern = os.path.join(dir_name, f"{name_part}_*{ext}")
                    existing_files = glob.glob(pattern)
                    if existing_files:
                        # Extract sequence numbers and find the highest
                        seq_nums = []
                        for file_path in existing_files:
                            filename = os.path.basename(file_path)
                            # Extract sequence number from filename like "name_000.jsonl"
                            try:
                                # Remove name_part and ext, extract number
                                seq_str = filename[len(name_part) + 1 : -len(ext) if ext else None]
                                seq_num = int(seq_str)
                                seq_nums.append((seq_num, file_path))
                            except (ValueError, IndexError):
                                continue
                        
                        if seq_nums:
                            # Sort by sequence number
                            seq_nums.sort()
                            # Process all files to collect processed IDs and find the last file
                            for seq_num, file_path in seq_nums:
                                try:
                                    with open(file_path, "r", encoding="utf-8") as f:
                                        file_count = 0
                                        for line in f:
                                            line = line.strip()
                                            if not line:
                                                continue
                                            try:
                                                obj = json.loads(line)
                                                if "summary" in obj:
                                                    continue
                                                rec_id = obj.get("id")
                                                if rec_id:
                                                    layerwise_processed_ids.add(rec_id)
                                                    file_count += 1
                                            except Exception:
                                                continue
                                        # Track the last file and its count
                                        if seq_num >= current_seq:
                                            current_seq = seq_num
                                            samples_in_file = file_count
                                except Exception:
                                    continue
                            
                            print(f"Resuming layerwise: found {len(layerwise_processed_ids)} samples in {len(seq_nums)} files. Last file: {get_layerwise_filename(current_seq)} with {samples_in_file} samples.")
                            log_line(f"Resuming layerwise: found {len(layerwise_processed_ids)} samples in {len(seq_nums)} files. Last file: {get_layerwise_filename(current_seq)} with {samples_in_file} samples.")
                            
                            # If last file is full (100 samples), move to next file
                            if samples_in_file >= 100:
                                current_seq += 1
                                samples_in_file = 0
                
                layerwise_writer_state = {
                    "base_dir": dir_name,
                    "name_part": name_part,
                    "ext": ext,
                    "get_filename": get_layerwise_filename,
                    "current_writer": None,
                    "current_seq": current_seq,
                    "samples_in_file": samples_in_file,
                    "all_writers": [],  # Track all opened files for cleanup
                }
                # Open the appropriate file (append if resuming to existing file, write if new)
                target_filename = get_layerwise_filename(current_seq)
                os.makedirs(dir_name, exist_ok=True)
                mode = "a" if args.resume and samples_in_file > 0 else "w"
                layerwise_writer_state["current_writer"] = open(target_filename, mode, encoding="utf-8")
                layerwise_writer_state["all_writers"].append(layerwise_writer_state["current_writer"])
            if args.layerwise_summary_only:
                layerwise_summary = {}

    total = overall_total
    for idx, sample in enumerate(samples, 1):
        sample_id = sample.get("id")
        if args.resume and sample_id:
            if sample_id in processed_success:
                continue
            if sample_id in processed_error and not args.resume_errors:
                continue
        prompt_kwargs = {
            "sample": sample,
            "custom_instructions": custom_user,
            "disallow_tools": args.no_tools,
        }
        try:
            import inspect

            if "hint" in inspect.signature(build_prompt).parameters:
                prompt_kwargs["hint"] = args.hint
        except Exception:
            pass
        user_prompt = build_prompt(**prompt_kwargs)
        full_prompt = (system_prompt + "\n\n" if system_prompt else "") + user_prompt
        eval_config = {
            "arch": args.arch,
            "model_params": args.model_params,
            "active_params": args.active_params,
            "diffusion_steps": args.diffusion_steps,
            "decoding": {
                "temperature": args.temperature,
                "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "stop": args.stop,
            "stop_patterns": args.stop_pattern,
        },
    }

        try:
            gen = provider.generate(user_prompt, system=system_prompt)
            layerwise_metrics = None
            if args.layerwise and args.provider == "transformers":
                metric_list = None
                if args.layerwise_metrics:
                    metric_list = [m.strip() for m in args.layerwise_metrics.split(",") if m.strip()]
                layerwise_metrics = provider.layerwise_metrics(
                    user_prompt,
                    system=system_prompt,
                    max_tokens=args.layerwise_max_tokens,
                    token_strategy=args.layerwise_token_strategy,
                    seed=args.layerwise_seed,
                    metrics=metric_list,
                )
                if layerwise_metrics and layerwise_summary is not None:
                    for key, values in layerwise_metrics.items():
                        if not isinstance(values, list):
                            continue
                        agg = layerwise_summary.setdefault(key, {"sum": [0.0] * len(values), "n": 0})
                        agg["sum"] = [a + float(b) for a, b in zip(agg["sum"], values)]
                        agg["n"] += 1

            rec = build_result_record(
                sample=sample,
                provider_name=gen.provider,
                model_name=gen.model,
                response_text=gen.content,
                usage=gen.usage,
                elapsed=gen.elapsed,
                prompt_text=full_prompt,
                eval_config=eval_config,
            )
            if args.log_raw_response:
                raw = gen.raw
                try:
                    if hasattr(raw, "to_dict"):
                        raw = raw.to_dict()
                    elif hasattr(raw, "model_dump"):
                        raw = raw.model_dump()
                except Exception:
                    pass
                try:
                    json.dumps(raw)
                    rec["raw_response"] = raw
                except Exception:
                    rec["raw_response"] = repr(raw)
            if layerwise_metrics is not None and args.layerwise_embed and not args.layerwise_summary_only:
                rec["layerwise"] = layerwise_metrics
            if layerwise_metrics is not None and layerwise_writer_state and not args.layerwise_summary_only:
                # Skip if this sample ID was already written to layerwise files
                sample_id = sample.get("id")
                if sample_id and sample_id in layerwise_processed_ids:
                    # Already processed, skip writing
                    pass
                else:
                    # Rotate to new file every 100 samples
                    if layerwise_writer_state["samples_in_file"] >= 100:
                        # Flush and close current file
                        layerwise_writer_state["current_writer"].flush()
                        layerwise_writer_state["current_writer"].close()
                        # Open next file
                        layerwise_writer_state["current_seq"] += 1
                        next_filename = layerwise_writer_state["get_filename"](layerwise_writer_state["current_seq"])
                        # Check if file exists (shouldn't normally, but handle resume case)
                        mode = "a" if os.path.exists(next_filename) else "w"
                        if mode == "a":
                            # Count existing samples in the file and collect IDs
                            try:
                                with open(next_filename, "r", encoding="utf-8") as f:
                                    existing_count = 0
                                    for line in f:
                                        line = line.strip()
                                        if not line:
                                            continue
                                        try:
                                            obj = json.loads(line)
                                            if "summary" in obj:
                                                continue
                                            rec_id = obj.get("id")
                                            if rec_id:
                                                layerwise_processed_ids.add(rec_id)
                                                existing_count += 1
                                        except Exception:
                                            continue
                                layerwise_writer_state["samples_in_file"] = existing_count
                            except Exception:
                                layerwise_writer_state["samples_in_file"] = 0
                        else:
                            layerwise_writer_state["samples_in_file"] = 0
                        layerwise_writer_state["current_writer"] = open(next_filename, mode, encoding="utf-8")
                        layerwise_writer_state["all_writers"].append(layerwise_writer_state["current_writer"])
                    
                    # Write to current file
                    layerwise_writer_state["current_writer"].write(
                        json.dumps({"id": sample.get("id"), "layerwise": layerwise_metrics}, ensure_ascii=False) + "\n"
                    )
                    layerwise_writer_state["current_writer"].flush()
                    layerwise_writer_state["samples_in_file"] += 1
                    # Track this ID as processed
                    if sample_id:
                        layerwise_processed_ids.add(sample_id)
        except Exception as e:
            rec = {
                "id": sample.get("id"),
                "error": str(e),
            }
            log_line(f"Error on {sample.get('id')}: {e}")

        # Append and persist incrementally
        results.append(rec)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        processed_count = resume_offset + idx
        if processed_count % 10 == 0 or processed_count == total:
            print(f"Progress: {processed_count}/{total} processed.")
            log_line(f"Progress: {processed_count}/{total} processed.")

    # Summary line as the final record
    summary = summarize([r for r in results if "judgement" in r])
    if layerwise_summary:
        layerwise_avg = {}
        for key, agg in layerwise_summary.items():
            if agg["n"] > 0:
                layerwise_avg[key] = [v / agg["n"] for v in agg["sum"]]
        summary["layerwise_summary"] = layerwise_avg
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")

    print("Done.")
    print("Summary:")
    print(json.dumps(summary, indent=2))
    log_line("Summary:")
    log_line(json.dumps(summary, ensure_ascii=False))

    if layerwise_writer_state:
        # Write summary to the last file if needed
        if layerwise_summary and layerwise_writer_state["current_writer"]:
            layerwise_writer_state["current_writer"].write(
                json.dumps({"summary": summary.get("layerwise_summary")}, ensure_ascii=False) + "\n"
            )
        # Close all opened files
        for writer in layerwise_writer_state["all_writers"]:
            if writer and not writer.closed:
                writer.close()


if __name__ == "__main__":
    run()
