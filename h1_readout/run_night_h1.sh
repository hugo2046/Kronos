#!/usr/bin/env bash
# H1 夜间编排（计划 §3.2→§3.3→§3.4 前半，20260905 H1 计划）
# 顺序：四臂训练（在线前向，错峰 16:30 守卫）→ 四臂 checkpoint 门禁 →
# 两窗打分落盘（不判读）→ DuckDB 追加对拍 → 引擎全表封存（不判读）。
# judge（一次开封）由会话手动执行。
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/user/miniconda3/envs/quant/bin/python
LOG_DIR=h1_readout/logs
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

for arm in H1a-lin H1a-kda H1b-lin H1b-kda; do
  if [ -f "h1_readout/data/${arm}_s42_best.pt" ]; then
    echo "[h1-pipeline] $(date '+%F %T') ${arm}_s42 checkpoint 已存在，跳过训练"
    continue
  fi
  echo "[h1-pipeline] $(date '+%F %T') 训练 ${arm}_s42（在线前向，~1.5-3h）"
  stagger_guard
  $PY -m h1_readout.train_h1 --arm "$arm" --seed 42 2>&1 | tee "$LOG_DIR/train_${arm}_s42.log"
done

echo "[h1-pipeline] $(date '+%F %T') 产物门禁：四臂 checkpoint + 四契约测试"
$PY -m pytest tests/test_h1_readout.py -v 2>&1 | tee "$LOG_DIR/test_gates.log"
for arm in H1a-lin H1a-kda H1b-lin H1b-kda; do
  test -f "h1_readout/data/${arm}_s42_best.pt" || { echo "缺 checkpoint ${arm}_s42" >&2; exit 1; }
done

echo "[h1-pipeline] $(date '+%F %T') 两窗打分落盘（不判读）"
stagger_guard
$PY -m h1_readout.run_h1_signals --seed 42 2>&1 | tee "$LOG_DIR/signals.log"

echo "[h1-pipeline] $(date '+%F %T') 四臂信号全部落盘 → DuckDB 追加对拍"
stagger_guard
$PY -m h1_readout.append_duckdb_h1 --seed 42 2>&1 | tee "$LOG_DIR/duckdb_append.log"

echo "[h1-pipeline] $(date '+%F %T') 引擎全表封存（不判读）"
stagger_guard
$PY -m h1_readout.run_h1_backtest --seed 42 2>&1 | tee "$LOG_DIR/backtest_seal.log"

echo "[h1-pipeline] $(date '+%F %T') 全部完成（judge 待会话一次开封）"
