import vllm_ascend
import vllm_ascend.utils

print("--- vllm_ascend members ---")
print([m for m in dir(vllm_ascend) if not m.startswith("__")])

print("\n--- vllm_ascend.utils members ---")
try:
    print([m for m in dir(vllm_ascend.utils) if not m.startswith("__")])
except:
    print("utils module not found")