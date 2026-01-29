import json
import random
from typing import Iterable, List, Optional


def \
    load_streetmath(
        dataset: str = "LuxMuseAI/StreetMathDataset",
        split: str = "test",
        shuffle: bool = False,
        seed: int = 42,
        limit: Optional[int] = None,
        local_jsonl: Optional[str] = None,
    ) -> List[dict]:
    """
    Load StreetMath dataset samples.

    - If `local_jsonl` is provided and exists, loads from it (JSONL with the same schema).
    - Else tries to load from Hugging Face Datasets via `datasets.load_dataset(dataset, split=split)`.

    Returns a list of dicts with keys including:
      id, topic, subtopic, prompt, split, exact_value.
    """
    records: List[dict] = []

    if local_jsonl:
        try:
            with open(local_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
        except FileNotFoundError:
            pass

    if not records:
        try:
            from datasets import load_dataset  # type: ignore

            ds = load_dataset(dataset, split=split)  # may need network
            # Convert to plain dicts for consistency
            records = [dict(x) for x in ds]
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset '{dataset}:{split}'. "
                f"If running offline, pass --local-jsonl to a local copy. Error: {e}"
            )

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(records)

    if limit is not None:
        records = records[: int(limit)]

    return records


def iter_chunks(items: List[dict], chunk_size: int) -> Iterable[List[dict]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _cuda_device_count() -> int:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        return 0
    return 0


def resolve_device_map(device_map: Optional[str], multi_gpu: bool = False) -> Optional[str]:
    """
    Resolve a transformers device_map value.

    - If device_map is explicitly provided, return it.
    - If multi_gpu is requested and multiple CUDA devices are available, return "auto".
    - Otherwise return None (caller should handle single-device placement).
    """
    if device_map:
        return device_map
    if not multi_gpu:
        return None
    if _cuda_device_count() > 1:
        return "auto"
    return None
