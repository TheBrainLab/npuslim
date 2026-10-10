import json, os, sys
sys.path.insert(0, "/llm_infer_workspace/ylt/npuslim/src")
sys.path.insert(0, "/llm_infer_workspace/ylt/glm52-qad/precompute")
os.environ.setdefault("HF_HOME", "/llm_infer_workspace/hf_home")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from precompute_logits import encode_one
from npuslim.qad.teacher_cache import TeacherCache

MODEL = "/llm_infer_workspace/ylt/npuslim/workspace/tiny_glm5"
CORPUS = ("/llm_infer_workspace/ylt/glm52-qad/glm52-qad-corpus/"
          "processed/corpus_v1_2026-08-27/clean")
CACHE = "/llm_infer_workspace/ylt/glm52-qad/precompute/logits_cache_tiny"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tc = TeacherCache(CACHE)
row0 = json.loads(open(f"{CORPUS}/A-long_horizon-tau-bench.jsonl").readline())
enc = encode_one(tok, row0, 1024)
ids = np.asarray(enc, dtype=np.int64)
print("enc len:", len(enc), "| sample id:", row0.get("id"))
fs = tc.fetch(row0["id"], expected_ids=np.asarray(enc[1:], dtype=np.int32))
tid = np.asarray(fs.topk_ids)
tlp = np.asarray(fs.topk_lps)
print("cache shape:", tid.shape, tlp.shape)
print("cache row0 col0-4 ids:", tid[0][:5].tolist())
print("cache row0 col0-4 lps:", [round(float(x), 3) for x in tlp[0][:5]])
print("cache col0 == actual(enc[1:]) ?",
      bool((tid[:len(enc)-1, 0] == np.asarray(enc[1:], dtype=tid.dtype)[:tid.shape[0]]).all()))

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
).to("npu:0").eval()
with torch.no_grad():
    out = model(input_ids=torch.from_numpy(ids).unsqueeze(0).to("npu:0"), use_cache=False)
lg = out.logits[0].float().cpu()
lp = torch.log_softmax(lg, dim=-1)

L = min(tid.shape[0], lg.shape[0] - 1)
m_top1 = 0
m_in100 = 0
true_lp_err = []
for i in range(L):
    t_ids = tid[i]
    true_tok = int(t_ids[0])
    t_top1 = int(t_ids[1])
    s_top1 = int(lg[i].argmax())
    m_top1 += (t_top1 == s_top1)
    m_in100 += (s_top1 in set(int(x) for x in t_ids[1:]))
    true_lp_err.append(abs(float(lp[i, true_tok]) - float(tlp[i, 0])))
print(f"rows={L} | student-argmax == teacher-top1: {m_top1}/{L} "
      f"| student-argmax in teacher-top100: {m_in100}/{L}")
print(f"|teacher_true_lp - student_true_lp| mean={np.mean(true_lp_err):.3f} "
      f"max={np.max(true_lp_err):.3f}")
for i in range(6):
    print(f"  row{i}: true_tok={int(tid[i,0])} t_lp={float(tlp[i,0]):.3f} "
          f"s_lp={float(lp[i, int(tid[i,0])]):.3f} | t_top1={int(tid[i,1])} "
          f"s_top1={int(lg[i].argmax())}")
print("student logits row0 top5:", torch.topk(lg[0], 5).indices.tolist(),
      [round(float(x), 2) for x in torch.topk(lg[0], 5).values.tolist()])
print("student lp row0 top5:", torch.topk(lp[0], 5).indices.tolist(),
      [round(float(x), 3) for x in torch.topk(lp[0], 5).values.tolist()])
