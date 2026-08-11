#!/usr/bin/env python3
"""Deployment pre-check for NPUSlim quantized checkpoints on vLLM-Ascend.

Validates that a quantized model directory is directly deployable with
`vllm serve --quantization ascend`, based on the lessons from the
GLM-5 W4A16 deployment (2026-08-10).

Checks performed:
  1. Must-FLOAT layers (kv_b_proj / indexer.wk / indexer.weights_proj):
     - quant_model_description.json marks them FLOAT
     - checkpoint only contains the raw .weight (no _scale/_offset tensors)
     - weight tensor dtype is floating point (bf16/fp16/fp32), not packed int32
  2. Every FLOAT entry in the desc must have no _scale/_offset keys in the index.
  3. Every quantized (non-FLOAT) weight in the desc must have its .weight key
     in the index (desc vs index consistency).
  4. Top-level desc keys required by vLLM-Ascend: model_quant_type / group_size.

Usage:
    python3 tools/check_model_deployable.py <model_dir> [--must-float kv_b_proj indexer.wk indexer.weights_proj]

Exit code 0 = OK, 1 = failures found.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

# safetensors canonical dtype names (lower-cased): BF16/F16/F32/F64/I8/I16/I32/I64/U8/U16/U32/U64/BOOL
FLOAT_DTYPES = {"bf16", "f16", "f32", "f64", "bfloat16", "float16", "float32", "float64"}
# Layers that vLLM-Ascend requires to stay FLOAT (see glm5_model.py).
DEFAULT_MUST_FLOAT = [
    "self_attn.kv_b_proj",
    "self_attn.indexer.wk",
    "self_attn.indexer.weights_proj",
]


def _read_safetensors_header(path: Path) -> dict:
    """Read the safetensors JSON header without loading tensor data."""
    with open(path, "rb") as f:
        nbytes = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(nbytes))
    return header


class Checker:
    def __init__(self, model_dir: Path, must_float: list[str]):
        self.dir = model_dir
        self.must_float = must_float
        self.index_path = model_dir / "model.safetensors.index.json"
        self.desc_path = model_dir / "quant_model_description.json"
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.index: dict | None = None
        self.desc: dict | None = None
        self._headers: dict[str, dict] = {}

    def _build_weight_map_from_shards(self) -> dict:
        wm: dict = {}
        for sf in sorted(self.dir.glob("*.safetensors")):
            try:
                header = _read_safetensors_header(sf)
            except Exception:
                continue
            for k in header.keys():
                if k != "__metadata__":
                    wm[k] = sf.name
        return wm

    def load(self) -> bool:
        if not self.dir.is_dir():
            self.errors.append(f"model dir not found: {self.dir}")
            return False
        self.desc = json.loads(self.desc_path.read_text()) if self.desc_path.is_file() else None
        if self.desc is None:
            self.warnings.append(
                "quant_model_description.json missing - quantized layers won't be recognized by vLLM-Ascend"
            )
        if self.index_path.is_file():
            self.index = json.loads(self.index_path.read_text())
        else:
            self.warnings.append("model.safetensors.index.json missing - falling back to scanning *.safetensors")
            self.index = {"weight_map": self._build_weight_map_from_shards()}
        return True

    def weight_map(self) -> dict:
        return self.index["weight_map"]

    def _tensor_info(self, key: str) -> dict | None:
        fname = self.weight_map().get(key)
        if fname is None:
            return None
        if fname not in self._headers:
            self._headers[fname] = _read_safetensors_header(self.dir / fname)
        return self._headers[fname].get(key)

    def check_must_float(self) -> None:
        wm = self.weight_map()
        for pat in self.must_float:
            suffix = "." + pat + ".weight"
            rel_keys = sorted(k for k in wm if k.endswith(suffix))
            if not rel_keys:
                self.warnings.append(f"no tensors found matching pattern: {pat}")
                continue
            for key in rel_keys:
                base = key[: -len(".weight")]
                # 1) no _scale/_offset tensors in index
                for aux in ("weight_scale", "weight_offset", "_scale", "_offset"):
                    if base + "." + aux in wm:
                        self.errors.append(f"[{pat}] {base}.{aux} still exists in checkpoint index")
                # 2) desc marks FLOAT
                if self.desc is not None:
                    dval = self.desc.get(key)
                    if dval is None:
                        self.warnings.append(f"[{pat}] {key}: not present in quant_model_description.json")
                    elif dval != "FLOAT":
                        self.errors.append(f"[{pat}] {key}: desc says {dval}, expected FLOAT")
                # 3) weight dtype is floating point
                info = self._tensor_info(key)
                if info is None:
                    self.errors.append(f"[{pat}] {key}: not found in shard header")
                else:
                    dtype = str(info["dtype"]).lower()
                    if dtype not in FLOAT_DTYPES:
                        self.errors.append(f"[{pat}] {key}: dtype {dtype} is not float (packed int4?)")

    def check_float_entries_have_no_aux(self) -> None:
        if self.desc is None:
            return
        wm = self.weight_map()
        for key, val in self.desc.items():
            if val != "FLOAT" or not key.endswith(".weight"):
                continue
            base = key[: -len(".weight")]
            for aux in ("weight_scale", "weight_offset", "_scale", "_offset"):
                if base + "." + aux in wm:
                    self.errors.append(f"[FLOAT entry] {base}.{aux} still exists in checkpoint index")

    def check_quantized_weights_present(self) -> None:
        if self.desc is None:
            return
        wm = self.weight_map()
        for key, val in self.desc.items():
            if val == "FLOAT" or not key.endswith(".weight"):
                continue
            if key not in wm:
                self.errors.append(f"[quantized] {key} ({val}) missing from checkpoint index")

    def check_desc_top_level(self) -> None:
        if self.desc is None:
            return
        if "model_quant_type" not in self.desc:
            self.errors.append("quant_model_description.json missing top-level key: model_quant_type")
        gs = self.desc.get("group_size")
        if gs is not None and int(gs) not in (32, 64, 128, 256):
            self.warnings.append(f"unusual group_size: {gs}")

    def run(self) -> None:
        if not self.load():
            return
        self.check_must_float()
        self.check_float_entries_have_no_aux()
        self.check_quantized_weights_present()
        self.check_desc_top_level()

    def report(self) -> None:
        print(f"checking: {self.dir}")
        nkeys = len(self.weight_map()) if self.index is not None else 0
        print(f"  desc: {'present' if self.desc is not None else 'MISSING'}, index keys: {nkeys}")
        for w in self.warnings:
            print(f"  [WARN ] {w}")
        for e in self.errors:
            print(f"  [ERROR] {e}")
        if not self.errors and not self.warnings:
            print("  OK: all checks passed")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", type=str, help="path to the quantized model directory")
    ap.add_argument("--must-float", nargs="*", help="layer patterns that must remain FLOAT (default from module)")
    ap.add_argument("--no-default-must-float", action="store_true", help="do not apply default must-FLOAT patterns")
    args = ap.parse_args(argv)

    must_float = DEFAULT_MUST_FLOAT if not args.no_default_must_float else []
    if args.must_float:
        must_float = args.must_float

    checker = Checker(Path(args.model_dir), must_float)
    checker.run()
    checker.report()
    print(f"result: {'OK' if not checker.errors else 'FAIL'}")
    return 1 if checker.errors else 0


if __name__ == "__main__":
    sys.exit(main())
