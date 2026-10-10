#!/bin/bash
# preflight_check.sh — 长任务发射前资源门（含 HBM 空闲门，见 0909 坑㊴）
# 用法：bash preflight_check.sh [每卡最小空闲 GiB]  默认 48（GLM-5.2 QAD 训练约 40G/卡）；量化传 12。
# 安全原则（★）：任何解析失败/读数不全（<8 卡）一律判 FAIL，绝不"看不懂=放行"。
set -u
MIN_FREE=${1:-48}
NODES=(${PREFLIGHT_NODES:-C3_super2 C3_super3 C3_super4 C3_super5})
FAIL=0
echo "== 发射前资源门（每卡需 >=${MIN_FREE}GiB 空闲）=="
for n in "${NODES[@]}"; do
  raw=$(ssh -o BatchMode=yes "$n" "npu-smi info 2>/dev/null" 2>/dev/null)
  used=$(echo "$raw" | grep -oE '[0-9]+ */ *65536' | grep -oE '^[0-9]+')
  cnt=$(echo "$used" | grep -c '[0-9]')
  hostav=$(ssh -o BatchMode=yes "$n" "free -g | awk '/^Mem:/{print \$7}'" 2>/dev/null)
  procs=$(ssh -o BatchMode=yes "$n" "docker exec e6_noawq_env pgrep -fc '[r]un.py' 2>/dev/null" 2>/dev/null | head -1)
  bad=""
  if [ "${cnt:-0}" -lt 8 ]; then
    bad="HBM 读数不全（${cnt:-0}/8 卡）→ 按 FAIL 处理"
  else
    i=0
    for u in $used; do
      fg=$(( (65536 - u) / 1024 ))
      if [ "$fg" -lt "$MIN_FREE" ]; then bad="$bad c$i(free=${fg}G)"; fi
      i=$((i + 1))
    done
  fi
  if [ -n "$bad" ]; then
    echo "[$n] FAIL ${bad}  [run.py=${procs:-?}, host=${hostav:-?}G]"
    FAIL=1
  elif [ "${procs:-0}" -gt 0 ]; then
    echo "[$n] WARN HBM 空闲够但已有 run.py 进程（${procs} 个）[host=${hostav:-?}G]"
    FAIL=1
  else
    echo "[$n] OK [host=${hostav:-?}G]"
  fi
done
if [ "$FAIL" -eq 0 ]; then echo "== 全部通过 =="; else echo "== 有阻塞项：先清理再发射 =="; fi
exit $FAIL
