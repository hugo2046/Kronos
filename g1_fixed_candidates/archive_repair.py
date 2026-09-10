"""历史 R 图片归档修复（20260910 计划 §6/§7，零回测零训练零推理）。

``ee77884`` 的 recorder ``a3b2656c…`` 用旧 ``R.save_objects(fig=Path)``
把 Path 对象 pickle 成产物，MLflow 侧没有真正 PNG 字节；``3dd3d6f`` 的
``save_figure_artifacts`` 已改为按文件上传。本脚本读取**已入 Git** 的四张
对照 PNG（与 ``outputs_sha256.json`` 冻结哈希逐字节核对），在新
audit recorder 下按文件重新上传并下载回读对拍，导出可验证 manifest。
不覆写任何历史记录、不重跑回测。

用法::

    python -m g1_fixed_candidates.archive_repair
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from g1_fixed_candidates.run import (  # noqa: E402
    EXPERIMENT, OUT_DIR, REPO_ROOT, save_figure_artifacts, sha256_file)

PARENT_RECORDER = "a3b2656c17b04deaa67d03bbc801b04d"   # ee77884 的 run
PARENT_EXPERIMENT = "986718269328784667"
SOURCE_COMMIT = "ee77884"          # PNG 字节入库提交
FIX_COMMIT = "3dd3d6f"             # 归档修复提交
REPAIR_EXPERIMENT = "kronos-g1-fixed-candidates-figure-repair"
FIGURES = ("fig_W3_main.png", "fig_W3_appendix.png",
           "fig_W4_main.png", "fig_W4_appendix.png")


def main() -> int:
    import os

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    # 1) 输入身份：与 ee77884 冻结的输出哈希清单逐字节核对（零回测）
    frozen = json.loads((OUT_DIR / "outputs_sha256.json").read_text(
        encoding="utf-8"))["files"]
    checks = {}
    for name in FIGURES:
        p = OUT_DIR / name
        assert p.is_file(), f"缺少已入Git图片：{p}"
        sha = sha256_file(p)
        key = name
        assert frozen[key]["sha256"] == sha, \
            f"{name} 与冻结哈希不一致（{sha}）"
        checks[name] = sha

    # 2) 新 audit recorder 上传（不触历史记录）。直连 MLflowRecorder
    #    （与回归测试同路径），不 qlib.init、不触 DDB。
    import mlflow
    from qlib.workflow.recorder import MLflowRecorder

    uri = (REPO_ROOT / "mlruns").as_uri()      # 仓库默认 mlruns 文件存储
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name(REPAIR_EXPERIMENT)
    exp_id = (exp.experiment_id if exp is not None
              else client.create_experiment(REPAIR_EXPERIMENT))
    run = client.create_run(exp_id, tags={
        "purpose": "figure_archive_repair",
        "parent_recorder": PARENT_RECORDER,
        "parent_experiment": PARENT_EXPERIMENT,
        "source_commit": SOURCE_COMMIT, "fix_commit": FIX_COMMIT,
        "backtest_rerun": "0", "training": "0", "inference": "0",
        "mlflow.runName": "figure_archive_repair",
    })
    rid = run.info.run_id
    client.log_param(rid, "files", json.dumps(list(FIGURES)))
    client.log_param(rid, "sha256", json.dumps(checks))
    rec = MLflowRecorder(exp_id, uri, mlflow_run=run)
    save_figure_artifacts(rec, OUT_DIR)

    # 3) 下载回读逐字节对拍
    stored = client.list_artifacts(rid, "figures")
    names_stored = {Path(i.path).name for i in stored}
    assert names_stored == set(FIGURES), \
        f"figures 产物不全：{names_stored}"
    readback = {}
    for item in stored:
        dl = Path(client.download_artifacts(rid, item.path))
        sha_dl = sha256_file(dl)
        assert sha_dl == checks[Path(item.path).name], \
            f"{item.path} 回读字节与源不一致"
        readback[Path(item.path).name] = {
            "mlflow_path": item.path, "downloaded_sha256": sha_dl,
            "byte_equal": True}
    client.set_terminated(rid, status="FINISHED")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "repaired",
        "repair_experiment": REPAIR_EXPERIMENT,
        "repair_experiment_id": exp_id,
        "repair_recorder_id": rid,
        "parent_recorder": PARENT_RECORDER,
        "parent_experiment_id": PARENT_EXPERIMENT,
        "source_commit": SOURCE_COMMIT, "fix_commit": FIX_COMMIT,
        "source_png_sha256": checks,
        "readback_verification": readback,
        "note": "旧 recorder 的 save_objects(fig=Path) 产物为 pickle 的 "
                "Path 对象非 PNG 字节；本 recorder 按文件重新上传 Git 内 "
                "PNG，未覆写历史记录、零回测零训练零推理。",
    }
    out = OUT_DIR / "figure_archive_repair_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    logger.info(f"[archive_repair] {rid} 上传 {len(FIGURES)} 图并回读对拍"
                f"通过；manifest → {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
