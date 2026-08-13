#!/bin/bash
# 阶段 3+4 链式执行：等阶段 2 链进程退出 → Stage 3 网格 → Stage 4 监督臂
# 用 Stage 2 链 PID 退出作为 GPU 空闲信号（比 JSON 更鲁棒：B3 失败也不卡死）。
set -u
PY=/home/user/miniconda3/envs/quant/bin/python
cd /home/user/workspace/Kronos
LOG=improve_suite/data/pipeline.log
STAGE2_PID="${1:-}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "==== Stage 3+4 chain 启动 ===="
if [ -n "$STAGE2_PID" ]; then
  log "等待 Stage 2 链进程 PID=$STAGE2_PID 退出（GPU 串行信号）..."
  while kill -0 "$STAGE2_PID" 2>/dev/null; do sleep 30; done
  log "Stage 2 链进程已退出，GPU 空闲"
fi

# Stage 3：三配置推理（C1 最短先跑）
for cfg in C1 C2 C3; do
  log "==== Stage 3: infer $cfg ===="
  $PY -m improve_suite.run_stage3_grid --infer $cfg >> "$LOG" 2>&1
  log "infer $cfg exit=$?"
done
log "==== Stage 3: engine + 判读 ===="
$PY -m improve_suite.run_stage3_grid --engine >> "$LOG" 2>&1
log "stage3 engine exit=$?"

# Stage 4：token 编码 → 训练 → 引擎
log "==== Stage 4: encode ===="
$PY -m improve_suite.run_stage4 --encode >> "$LOG" 2>&1
log "encode exit=$?"
log "==== Stage 4: train + 闸门 ===="
$PY -m improve_suite.run_stage4 --train >> "$LOG" 2>&1
log "train exit=$?"
log "==== Stage 4: engine ===="
$PY -m improve_suite.run_stage4 --engine >> "$LOG" 2>&1
log "engine exit=$?"

log "==== Stage 3+4 chain 完成 ===="
ls -la improve_suite/data/stage3_*.json improve_suite/data/stage4_*.json 2>/dev/null >> "$LOG" 2>&1
