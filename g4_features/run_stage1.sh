#!/bin/bash
# G4 阶段 1 串行链（G4 计划 §3）：三种子 tokenizer→predictor→两窗推理。
# 训练幂等：summary.json 存在即跳过（该文件只在训练正常完成后写出）；
# 推理幂等：run_g4_signals 自带断点续跑。每步 console 落盘 g4_features/data/。
set -u
PY=/home/user/miniconda3/envs/quant/bin/python
cd /home/user/workspace/Kronos

train() {  # seed stage
  local seed=$1 stage=$2
  local dir="g4_features/outputs/models/finetune_${stage}_g4_s${seed}"
  if [ -f "$dir/summary.json" ]; then echo "[skip-train] $dir 已完成"; return 0; fi
  echo "[train] seed=$seed stage=$stage start=$(date +%F_%T)"
  "$PY" g4_features/train.py --seed "$seed" --stage "$stage" \
    > "g4_features/data/train_${stage}_g4_s${seed}_console.txt" 2>&1 \
    || { echo "[FAIL-train] seed=$seed stage=$stage（见 console）"; exit 1; }
  echo "[train] seed=$seed stage=$stage done=$(date +%F_%T)"
}

infer() {  # seed window
  local seed=$1 w=$2
  mkdir -p "g4_features/data/s${seed}"
  echo "[infer] seed=$seed window=$w start=$(date +%F_%T)"
  "$PY" g4_features/run_g4_signals.py --seed "$seed" --window "$w" \
    > "g4_features/data/s${seed}/infer_${w}_console.txt" 2>&1 \
    || { echo "[FAIL-infer] seed=$seed window=$w（见 console）"; exit 1; }
  echo "[infer] seed=$seed window=$w done=$(date +%F_%T)"
}

for seed in 100 101 102; do
  train "$seed" tokenizer
  train "$seed" predictor
done

for seed in 100 101 102; do
  for w in backtest 2025h2; do
    infer "$seed" "$w"
  done
done

echo "STAGE1_ALL_DONE $(date +%F_%T)"
