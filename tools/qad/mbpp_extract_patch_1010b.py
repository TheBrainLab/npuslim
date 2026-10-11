#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXTRACT_PATCH_1010B — mbpp 判据方言提取固化件（★ 1011 入仓）。

背景：GLM/CodeLlama 系 0-shot plain mbpp 会说 [PYTHON]...[/PYTHON] 方言，
plain mbpp 的 utils.py 无提取滤镜 → pass@1 塌到 ~0.02（与 ppl 数量级异常同级的
"配置/提取链错信号"）。对策 = 优先级提取：最后一个 [PYTHON] 块 > 最后一个
markdown 围栏 > 原文回退。

已在 e6_noawq_env 容器（super2/3/4/5）lm_eval 路径实测有效：
  /usr/local/python3.12.13/lib/python3.12/site-packages/lm_eval/tasks/mbpp/utils.py
  → pass_at_1() 内 predictions 二次包装 _extract_mbpp_code_1010b(p)。

容器是 ephemeral（docker rm/recreate 或镜像重 build 即丢），本件为源码归档；
镜像重建后用 apply(utils_path) 重打。读法：
  from tools.qad.mbpp_extract_patch_1010b import apply
  apply("/usr/local/python3.12.13/lib/python3.12/site-packages/lm_eval/tasks/mbpp/utils.py")
"""
import re

PATCH_FN_SRC = '''def _extract_mbpp_code_1010b(text):
    """EXTRACT_PATCH_1010B: 提取优先级：最后一个 [PYTHON] 块 > 最后一个 markdown 围栏 > 原文。
    （GLM/CodeLlama 系模型 0-shot plain mbpp 会说 [PYTHON]...[/PYTHON] 方言。）"""
    import re as _re
    mk = _re.findall(r"\\[PYTHON\\]\\n?(.*?)\\n?\\[/PYTHON\\]", text, _re.DOTALL)
    if mk and mk[-1].strip():
        return mk[-1]
    md = _re.findall(r"```(?:[a-zA-Z0-9_+\\-]*)[ \\t]*\\n?(.*?)```", text, _re.DOTALL)
    if md and md[-1].strip():
        return md[-1]
    return text
'''

PATCH_CALL_SRC = """        # EXTRACT_PATCH_1010: GLM 输出带 markdown 围栏/[PYTHON] 方言，plain mbpp 不做提取 →
        # 先剥方言再执行（提取失败回退原文）
        predictions = [[(_extract_mbpp_code_1010b(p) or p) for p in pred] for pred in predictions]
"""


def apply(utils_path: str) -> bool:
    """向 mbpp/utils.py 注入 patch（幂等）。成功返回 True；已打标记返回 False。"""
    src = open(utils_path).read()
    if "_extract_mbpp_code_1010b" in src:
        return False  # already patched
    needle = "    if isinstance(predictions[0], str):\n        predictions = [[p] for p in predictions]\n"
    if needle not in src:
        raise RuntimeError("anchor not found in mbpp/utils.py: pass_at_1 body changed")
    body = needle + PATCH_CALL_SRC.rstrip("\n") + "\n"
    src = src.replace(needle, body, 1)
    src = src.rstrip("\n") + "\n\n\n" + PATCH_FN_SRC.rstrip("\n") + "\n"
    open(utils_path, "w").write(src)
    return True


def extract(text: str) -> str:
    """与容器内 _extract_mbpp_code_1010b 行为一致的纯函数（重评/离线用）。"""
    mk = re.findall(r"\[PYTHON\]\n?(.*?)\n?\[/PYTHON\]", text, re.DOTALL)
    if mk and mk[-1].strip():
        return mk[-1]
    md = re.findall(r"```(?:[a-zA-Z0-9_+\-]*)[ \t]*\n?(.*?)```", text, re.DOTALL)
    if md and md[-1].strip():
        return md[-1]
    return text


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print("patched:", apply(sys.argv[1]))
    else:
        # 自测
        t1 = "[PYTHON]\ndef f(x):\n    return x\n[/PYTHON]"
        t2 = "Sure, here:\n```python\ndef g():\n    pass\n```"
        assert extract(t1) == "def f(x):\n    return x"
        assert extract(t2).strip() == "def g():\n    pass"
        assert extract("plain answer") == "plain answer"
        print("selftest OK")
