import json

import torch

from npuslim.datasets.text_dataset import TextDataset


class DummyProcessor:
    bos_token = "<bos>"
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        _ = tokenize, add_generation_prompt
        return "\n".join(f"{m['role']}:{m['content']}" for m in messages)

    def __call__(self, texts, return_tensors, max_length, truncation, padding):
        _ = return_tensors, truncation, padding
        batch = len(texts)
        input_ids = torch.arange(max_length).unsqueeze(0).repeat(batch, 1)
        attn = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attn}


def test_text_dataset_load_jsonl(tmp_path):
    data_path = tmp_path / "calib.jsonl"
    sample = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
    }
    data_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    ds = TextDataset(
        processor=DummyProcessor(),
        data_path=str(data_path),
        num_samples=1,
        max_seq_length=16,
        device="cpu",
    )

    assert len(ds) == 1
    item = ds[0]
    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}
    assert item["input_ids"].shape == (1, 16)
    assert item["attention_mask"].shape == (1, 16)
    assert item["labels"].shape == (1, 16)
    assert item["labels"][0, -1].item() == -100
