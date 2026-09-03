#!/usr/bin/env bash
# R1 夜间编排（计划 §4.2，20260903 L1与R1计划）
# 顺序：day_groups+六训（IC 损失）→ 6 checkpoint 门禁 → 两窗打分落盘 →
# 引擎全表封存（不判读）。judge（一次开封）由会话手动执行。
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/user/miniconda3/envs/quant/bin/python
LOG_DIR=r1_objective/logs
mkdir -p "$LOG_DIR"

# —— 16:30 登记 cron 错峰守卫：16:25~16:50 CST 内不动 GPU（等 cron 跑完）——
stagger_guard() {
  local hm
  hm=$(date +%H%M)
  if [ "$hm" -ge 1625 ] && [ "$hm" -le 1650 ]; then
    echo "[stagger] $hm 落在 16:30 登记 cron 时段，sleep 至 16:51" >&2
    sleep $(( (1651 - 10#$hm) * 60 ))
  fi
}

echo "[r1-pipeline] $(date '+%F %T') 4.2 六训启动（IC 损失；首跑含 day_groups 构建对拍）"
stagger_guard
$PY -m r1_objective.run_r1_train 2>&1 | tee "$LOG_DIR/train.log"

echo "[r1-pipeline] $(date '+%F %T') 产物门禁：六 checkpoint + day_groups 完整性"
$PY -m pytest tests/test_l1_r1.py::test_r1_day_groups_cache -v 2>&1 | tee "$LOG_DIR/test_gates.log"
for arm in R-lin R-kda; do
  for seed in 42 43 44; do
    test -f "r1_objective/data/${arm}_s${seed}_best.pt" || { echo "缺 checkpoint ${arm}_s${seed}" >&2; exit 1; }
  done
done

echo "[r1-pipeline] $(date '+%F %T') 两窗打分落盘（不判读）"
stagger_guard
$PY -m r1_objective.run_r1_signals 2>&1 | tee "$LOG_DIR/signals.log"

echo "[r1-pipeline] $(date '+%F %T') 引擎全表封存（不判读）"
stagger_guard
$PY -m r1_objective.run_r1_backtest 2>&1 | tee "$LOG_DIR/backtest_seal.log"

echo "[r1-pipeline] $(date '+%F %T') 全部完成（judge 待会话一次开封）"
