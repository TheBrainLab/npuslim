from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from loguru import logger

from npuslim.datasets.base_dataset import BaseDataset
from npuslim.core import DatasetRegistry


@DatasetRegistry.register("Text", aliases=["TextDataset", "text"])
class TextDataset(BaseDataset):
    """Text calibration dataset from JSONL or Parquet files."""

    def __init__(self, *args, data_path: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_path = str(data_path)
        self._load_data()

    def _load_data(self) -> None:
        lower_path = self.data_path.lower()
        if lower_path.endswith(".parquet"):
            self._load_parquet_data()
        else:
            self._load_jsonl_data()

    def _load_parquet_data(self) -> None:
        try:
            import pyarrow.parquet as pq
        except Exception as exc:
            raise ImportError(
                "TextDataset parquet mode requires pyarrow. Install with: pip install pyarrow"
            ) from exc

        table = pq.read_table(self.data_path)
        df = table.to_pandas()
        total_samples = min(self.num_samples, len(df)) if self.num_samples > 0 else len(df)

        for i in range(total_samples):
            text = df["text"].iloc[i]
            model_inputs = self.processor(
                [text],
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )

            if "labels" in df.columns:
                labels = torch.tensor(df["labels"].iloc[i]).unsqueeze(0)
            else:
                labels = model_inputs["input_ids"].roll(shifts=-1, dims=-1)
                labels[:, -1] = -100

            self.data.append(
                {
                    "input_ids": model_inputs["input_ids"].to(self.device),
                    "attention_mask": model_inputs["attention_mask"].to(self.device),
                    "labels": labels.to(self.device),
                }
            )

    def _load_jsonl_data(self) -> None:
        line_count = 0
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if self.num_samples > 0 and line_count >= self.num_samples:
                    break

                data = json.loads(line)
                if not ("messages" in data or "input" in data or "conversations" in data):
                    raise ValueError("JSON format error: missing messages/input/conversations")

                messages = self._prepare_messages(data)
                text = self._build_text(messages)

                model_inputs = self.processor(
                    [text],
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                )

                labels = model_inputs["input_ids"].roll(shifts=-1, dims=-1)
                labels[:, -1] = -100

                self.data.append(
                    {
                        "input_ids": model_inputs["input_ids"].to(self.device),
                        "attention_mask": model_inputs["attention_mask"].to(self.device),
                        "labels": labels.to(self.device),
                    }
                )
                line_count += 1

        logger.info(f"Loaded TextDataset from {self.data_path}: {len(self.data)} samples")

    def _build_text(self, messages: List[Dict[str, Any]]) -> str:
        supports_chat_template = (
            hasattr(self.processor, "apply_chat_template")
            and getattr(self.processor, "chat_template", None) is not None
        )
        if supports_chat_template:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
            )

        thinking_data = False
        for item in messages:
            if item.get("role") == "assistant":
                content = str(item.get("content", ""))
                if "<think>" in content and "</think>" in content:
                    thinking_data = True
                    break

        if not thinking_data:
            return text

        bos_token = getattr(self.processor, "bos_token", "") or ""
        eos_token = getattr(self.processor, "eos_token", "") or ""
        text = bos_token
        for item in messages:
            role = item.get("role")
            content = item.get("content", "")
            if role == "system":
                text += content
            elif role == "user":
                text += "<｜User｜>" + content + "<｜Assistant｜>"
            elif role == "assistant":
                text += content + eos_token
        return text

    def _prepare_messages(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        if "messages" in data:
            messages = list(data["messages"])
            system_prompt = data.get("system_prompt")
            if system_prompt and messages and messages[0].get("role") != "system":
                messages = [{"role": "system", "content": system_prompt}] + messages
        elif "conversations" in data:
            conv = data["conversations"]
            messages = [
                {"role": "user", "content": conv[0].get("value", "")},
                {"role": "assistant", "content": conv[1].get("value", "")},
            ]
            system_prompt = data.get("system_prompt") or data.get("system")
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            messages = [
                {"role": "user", "content": data.get("input", "")},
                {"role": "assistant", "content": data.get("output", "")},
            ]
            system_prompt = data.get("system_prompt")
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages

        for item in messages:
            if "role" not in item and "from" in item:
                item["role"] = item["from"]
            if "content" not in item and "value" in item:
                item["content"] = item["value"]
            role = str(item.get("role", "")).lower()
            if "human" in role:
                item["role"] = "user"
            elif "gpt" in role:
                item["role"] = "assistant"

        return messages
