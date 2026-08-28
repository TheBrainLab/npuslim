"""Mixed-domain calibration dataset (chat + code + math + multilingual).

Reference: msmodelSlim lab_calib/ practice -- multiple single-domain JSONL files
(mix_calib.jsonl / autocodebench.jsonl / cn_en.jsonl ...) mixed for calibration,
empirically better than pure C4/Wiki (APEX; dev plan v2 P1-3).

Accepted per-line formats (auto-detected):
  {"messages": [{"role": ..., "content": ...}, ...]}   # chat -> apply_chat_template
  {"text": "..."}                                      # raw text
  {"inputs_pretokenized": "..."}                        # msModelSlim lab_calib format

Sampling: files are interleaved round-robin (or by `weights`), then the merged
stream is tokenized and window-cropped exactly like C4Dataset (samples shorter
than max_seq_length are skipped -- GPTQ calibration requires equal-length batches
without padding).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from npuslim.datasets.base_dataset import BaseDataset
from npuslim.core import DatasetRegistry


@DatasetRegistry.register("MixedDomain", aliases=["Mixed", "MixedDomainDataset"])
class MixedDomainDataset(BaseDataset):
    """Multi-file mixed-domain calibration dataset (JSONL)."""

    def __init__(
        self,
        *args,
        data_path: Optional[str] = None,
        data_paths: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not data_paths:
            if not data_path:
                raise ValueError("MixedDomainDataset requires data_path or data_paths")
            data_paths = [data_path]
        self.data_paths = [str(p) for p in data_paths]
        self.weights = list(weights) if weights else None
        self.seed = int(seed)
        self._load_data()

    # ------------------------------------------------------------------ loading

    def _iter_file_lines(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    def _extract_text(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract calibration text from one JSONL record (any supported format)."""
        if "messages" in data or "conversations" in data:
            messages = self._prepare_messages(data)
            return self._render_chat(messages)
        for key in ("text", "inputs_pretokenized", "input", "prompt"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None

    def _prepare_messages(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        if "messages" in data:
            messages = list(data["messages"])
        else:  # conversations (sharegpt style: from/value)
            messages = [
                {"role": item.get("from", "user"), "content": item.get("value", "")}
                for item in data["conversations"]
            ]
        for item in messages:
            if "role" not in item and "from" in item:
                item["role"] = item["from"]
            if "content" not in item and "value" in item:
                item["content"] = item["value"]
            role = str(item.get("role", "")).lower()
            if "human" in role:
                item["role"] = "user"
            elif "gpt" in role or "assistant" in role:
                item["role"] = "assistant"
        return messages

    def _render_chat(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        supports_chat_template = (
            hasattr(self.processor, "apply_chat_template")
            and getattr(self.processor, "chat_template", None) is not None
        )
        if supports_chat_template:
            try:
                return self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
        )

    def _collect_texts(self) -> List[str]:
        """Collect and interleave texts from all files (round-robin or weighted)."""
        per_file: List[List[str]] = []
        for path in self.data_paths:
            texts: List[str] = []
            for data in self._iter_file_lines(path):
                text = self._extract_text(data)
                if text:
                    texts.append(text)
            per_file.append(texts)
            logger.info(
                f"MixedDomain: {Path(path).name} -> {len(texts)} usable samples"
            )

        weights = self.weights
        if weights and len(weights) == len(per_file):
            # Weighted interleave: each file contributes proportionally.
            merged: List[str] = []
            iterators = [iter(t) for t in per_file]
            active = list(range(len(per_file)))
            while active:
                for i in list(active):
                    take = max(1, round(weights[i]))
                    for _ in range(take):
                        item = next(iterators[i], None)
                        if item is None:
                            active.remove(i)
                            break
                        merged.append(item)
            return merged

        # Round-robin interleave (default): preserves domain balance regardless
        # of file sizes better than concatenation.
        merged = []
        iterators = [iter(t) for t in per_file]
        active = list(range(len(per_file)))
        while active:
            for i in list(active):
                item = next(iterators[i], None)
                if item is None:
                    active.remove(i)
                else:
                    merged.append(item)
        return merged

    def _load_data(self) -> None:
        random.seed(self.seed)
        texts = self._collect_texts()
        logger.info(
            f"MixedDomain: {len(texts)} interleaved samples from "
            f"{len(self.data_paths)} file(s), requesting {self.num_samples}"
        )

        count = 0
        for text in texts:
            if count >= self.num_samples:
                break

            trainenc = self.processor(text, return_tensors="pt")

            if trainenc.input_ids.shape[1] <= self.max_length:
                # GPTQ calibration needs fixed-length batches without padding;
                # skip short samples (same policy as C4Dataset).
                continue

            i = random.randint(0, trainenc.input_ids.shape[1] - self.max_length - 1)
            j = i + self.max_length
            inp = trainenc.input_ids[:, i:j]

            labels = inp.clone()
            labels[:, :-1] = -100

            self.data.append(
                {
                    "input_ids": inp.to(self.device),
                    "attention_mask": torch.ones_like(inp).to(self.device),
                    "labels": labels.to(self.device),
                }
            )
            count += 1

        if not self.data:
            logger.warning(
                "MixedDomain loaded successfully but no sample met max_seq_length "
                "requirement; all samples were shorter than the window"
            )
        else:
            logger.info(f"MixedDomain: {len(self.data)} samples ready")
