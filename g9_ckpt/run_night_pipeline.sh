#!/usr/bin/env bash
# G9 整夜编排（计划 §4.2→§4.3→§4.4 前半，20260821 计划）
# 顺序：等训练完成 → test_all_epoch_saver → 7 次推理（错峰 16:30 cron）→
# DuckDB 追加对拍 → 引擎全表封存。judge（一次开封）由会话手动执行。
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/user/miniconda3/envs/quant/bin/python
LOG_DIR=g9_ckpt/logs
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

echo "[pipeline] $(date '+%F %T') 等待 4.2 训练完成…"
for i in $(seq 1 720); do
  if grep -q "Training finished" "$LOG_DIR/train.log" 2>/dev/null; then
    echo "[pipeline] $(date '+%F %T') 训练完成"; break
  fi
  sleep 60
done
grep -q "Training finished" "$LOG_DIR/train.log"

echo "[pipeline] $(date '+%F %T') 4.2 产物门禁：test_all_epoch_saver"
$PY -m pytest tests/test_g9_ckpt.py::test_all_epoch_saver -v 2>&1 | tee "$LOG_DIR/test_saver.log"

# —— 4.3 七次推理（臂×窗，计划 §1 臂表；断点续跑幂等）——
run_signals() {
  local arm=$1 window=$2
  stagger_guard
  echo "[pipeline] $(date '+%F %T') 推理 arm=$arm window=$window"
  $PY -m g9_ckpt.run_g9_signals --arm "$arm" --window "$window" \
    >> "$LOG_DIR/signals_${arm}_${window}.log" 2>&1
}

run_signals E1 backtest
run_signals E1 2025h2
run_signals E15 backtest
run_signals E15 2025h2
run_signals E5 backtest
run_signals E10 backtest
run_signals E0 backtest

echo "[pipeline] $(date '+%F %T') 六组信号全部落盘 → DuckDB 追加对拍"
stagger_guard
$PY -m g9_ckpt.append_duckdb_g9 2>&1 | tee "$LOG_DIR/duckdb_append.log"

echo "[pipeline] $(date '+%F %T') 引擎全表封存（不判读）"
stagger_guard
$PY -m g9_ckpt.run_g9_backtest 2>&1 | tee "$LOG_DIR/backtest.log"

echo "[pipeline] $(date '+%F %T') 全部完成（judge 待会话一次开封）"
