#!/bin/bash
# 阶段 2 链式执行：等 canonical 路径完成 → 分布信号+R3 → B3 csi500
# 由 nohup 后台运行；每步日志追加到 pipeline.log。
set -u
PY=/home/user/miniconda3/envs/quant/bin/python
cd /home/user/workspace/Kronos
LOG=improve_suite/data/pipeline.log
PATHS_PID="${1:-}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "==== Stage 2 chain 启动 ===="

# 1. 等 canonical 路径进程结束（或 parquet 齐全）
if [ -n "$PATHS_PID" ]; then
  log "等待 canonical 路径进程 PID=$PATHS_PID ..."
  while kill -0 "$PATHS_PID" 2>/dev/null; do sleep 30; done
  log "canonical 路径进程已退出"
else
  log "未给 PID，轮询 oos parquet 是否齐全"
fi

# 2. 确认两窗 parquet 齐全 + 全量对拍门禁
log "==== 全量对拍门禁 ===="
$PY -m improve_suite.run_canonical_paths --gate-only --window paper >> "$LOG" 2>&1
$PY -m improve_suite.run_canonical_paths --gate-only --window oos >> "$LOG" 2>&1

# 3. 分布信号 S1~S3 + R3 回填
log "==== Stage 2: 分布信号 + R3 ===="
$PY -m improve_suite.run_stage2_dist >> "$LOG" 2>&1
log "stage2_dist exit=$?"

# 4. B3 csi500 跨池验证
log "==== Stage 2: B3 csi500 ===="
$PY -m improve_suite.run_b3_csi500 >> "$LOG" 2>&1
log "b3_csi500 exit=$?"

log "==== Stage 2 chain 完成 ===="
ls -la improve_suite/data/stage2_*.json improve_suite/figures/*.png 2>/dev/null >> "$LOG" 2>&1
