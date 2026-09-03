#!/usr/bin/env bash
# L1 夜间编排（计划 §4.3，20260903 L1与R1计划）
# 顺序：数据窗重建（幂等）→ L250-ft 训练（幂等）→ 八次推理（臂×窗，断点续跑）→
# DuckDB 追加对拍 → 引擎全表封存（不判读）。judge（一次开封）由会话手动执行。
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/user/miniconda3/envs/quant/bin/python
LOG_DIR=l1_context/logs
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

# —— 0. 数据窗重建（CPU/DDB，无 GPU；幂等跳过）——
if [ -f l1_context/data/ashares_lb250/train_data.pkl ]; then
  echo "[l1-pipeline] $(date '+%F %T') 数据窗已存在，跳过重建"
else
  echo "[l1-pipeline] $(date '+%F %T') lookback=250 数据窗重建"
  $PY -m l1_context.train_l250ft dataset 2>&1 | tee "$LOG_DIR/dataset_build.log"
fi

echo "[l1-pipeline] $(date '+%F %T') 产物门禁：test_l250ft_dataset"
$PY -m pytest tests/test_l1_r1.py::test_l250ft_dataset -v 2>&1 | tee "$LOG_DIR/test_dataset_gate.log"

# —— 1. L250-ft 训练（G1 配方 seed=100，CE 早停；幂等跳过）——
if [ -f l1_context/outputs/models/finetune_predictor_l250ft/checkpoints/best_model/config.json ]; then
  echo "[l1-pipeline] $(date '+%F %T') L250-ft checkpoint 已存在，跳过训练"
else
  echo "[l1-pipeline] $(date '+%F %T') L250-ft predictor 训练（epochs=15，CE 早停）"
  stagger_guard
  $PY -m l1_context.train_l250ft train 2>&1 | tee "$LOG_DIR/train_l250ft.log"
fi

# —— 2. 八次推理（臂×窗，计划 §1 臂表冻结；断点续跑幂等）——
run_signals() {
  local arm=$1 window=$2
  stagger_guard
  echo "[l1-pipeline] $(date '+%F %T') 推理 arm=$arm window=$window"
  $PY -m l1_context.run_l1_signals --arm "$arm" --window "$window" \
    >> "$LOG_DIR/signals_${arm}_${window}.log" 2>&1
}

run_signals L250ZS100 backtest
run_signals L250ZS101 backtest
run_signals L250ZS102 backtest
run_signals L500ZS100 backtest
run_signals L250FT100 backtest
run_signals L250ZS100 2025h2
run_signals L250ZS101 2025h2
run_signals L250ZS102 2025h2

echo "[l1-pipeline] $(date '+%F %T') 八组信号全部落盘 → DuckDB 追加对拍"
stagger_guard
$PY -m l1_context.append_duckdb_l1 2>&1 | tee "$LOG_DIR/duckdb_append.log"

echo "[l1-pipeline] $(date '+%F %T') 引擎全表封存（不判读）"
stagger_guard
$PY -m l1_context.run_l1_backtest 2>&1 | tee "$LOG_DIR/backtest_seal.log"

echo "[l1-pipeline] $(date '+%F %T') 全部完成（judge 待会话一次开封）"
