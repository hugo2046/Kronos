"""阶段 0 探测（计划 §4.3 先行项）。

三项，全部 PIT、不涉及窗口内信号数字：

1. ``sw_l1`` 行业数据可得性（决定 analyzer.analyze_factor 的 industry 参数）；
2. ``up_down_limit_status`` 取值语义（抽样对照 limit 价格字段验证 ±1/0 含义）；
3. 每档样本量与 ST 占比统计（结构性，跨全窗口月末）。

连接纪律（计划 §1.2）：显式 URI、mask_uri 脱敏、任何输出不得明文凭据。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# 引入本包（确保 sys.path 含仓库根）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronos_qlib import QlibProvider  # noqa: E402
from liquidity_strat.common import (  # noqa: E402
    DATA_DIR,
    DATA_END,
    LiquidityConfig,
    WINDOW_END,
    WINDOW_START,
    ensure_dirs,
    init_analyzer_auth,
    mask_uri,
)
from liquidity_strat.stratify import (  # noqa: E402
    _avg_amount_pct,
    _st_codes_at,
    month_end_rebalance_dates,
    stratify,
)


def probe_connection() -> str:
    """显式 URI auth + QlibProvider 连通性（脱敏输出）。"""
    uri = init_analyzer_auth()  # 内部已 mask
    logger.info(f"连接探测：analyzer.auth OK，URI={mask_uri(uri)}")
    return uri


def probe_industry_sw_l1(cfg: LiquidityConfig) -> bool:
    """探测 sw_l1 行业字段在 DDB 实例的可得性。

    尝试用 qlib 直接取 ``$sw_l1`` 字段；返回 True/False 并记录。
    不可用时，下游 analyzer.analyze_factor 改用 industry=None（计划 §0 空隙提醒）。
    """
    from qlib.data import D

    # 跨多日 / 多股抽样，避免单日单股偶然；csi300 成分覆盖各行业
    members = D.list_instruments(
        D.instruments("csi300"), start_time="2025-06-30", end_time="2025-06-30", as_list=True
    )
    sample = members[:30]
    logger.info(f"sw_l1 探测：抽样 {len(sample)} 只 csi300 成员，跨 2025-06 ~ 2025-07")
    for field in ("$sw_l1", "$sw_l1_code", "$industry_sw"):
        try:
            df = D.features(
                sample,
                [field],
                start_time="2025-06-01",
                end_time="2025-07-31",
            )
            if field not in df.columns:
                logger.info(f"  字段 {field}: 不存在")
                continue
            col = df[field]
            n_total = len(col)
            n_filled = col.notna().sum()
            # 关键：全部为同一常数（如 0.0）等于无行业区分力
            n_unique = col.dropna().nunique()
            logger.info(
                f"  字段 {field}: 非空 {n_filled}/{n_total}，唯一值数 {n_unique}"
            )
            if n_unique > 1 and n_filled > 0:
                logger.info(f"    取值示例: {sorted(col.dropna().unique())[:8]}")
                return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"  字段 {field}: 读取失败 {type(exc).__name__}: {exc}")
    logger.warning("sw_l1 行业字段在本实例无区分力（全空或全常数）→ analyze_factor 将用 industry=None")
    return False


def probe_up_down_limit_status(cfg: LiquidityConfig) -> dict:
    """探测 up_down_limit_status 取值语义（抽样对照 limit 价格）。

    取一段含涨跌停的样本区间，统计字段取值分布，并抽样核对：
    - 涨停日（close ≈ high_limit）→ 期望 status == 1
    - 跌停日（close ≈ low_limit）→ 期望 status == -1
    - 否则 → 期望 status == 0
    """
    from qlib.data import D

    sample_codes = D.list_instruments(
        D.instruments("csi300"), start_time="2025-01-01", end_time="2025-01-31", as_list=True
    )[:20]
    fields = ["$close", "$high_limit", "$low_limit", "$up_down_limit_status", "$limit"]
    try:
        df = D.features(
            sample_codes,
            fields,
            start_time="2025-01-01",
            end_time="2025-03-31",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"up_down_limit_status 取数失败：{type(exc).__name__}: {exc}")
        return {"available": False}

    # D.features 保留 `$` 前缀；统一去掉便于后续引用
    df = df.rename(columns={c: c.lstrip("$") for c in df.columns})
    cols_present = [c for c in df.columns]
    logger.info(f"up_down_limit_status 探测：字段 {cols_present}")
    if "up_down_limit_status" not in df.columns:
        logger.error("up_down_limit_status 字段缺失")
        return {"available": False}

    vc = df["up_down_limit_status"].value_counts(dropna=False).to_dict()
    logger.info(f"  取值分布（抽样）: {vc}")

    # 对照 limit 价格：close 与 high/low_limit 的关系
    result = {"available": True, "value_counts": {str(k): int(v) for k, v in vc.items()}}
    if {"high_limit", "low_limit", "close"}.issubset(df.columns):
        df = df.dropna(subset=["close", "high_limit", "low_limit", "up_down_limit_status"])
        if len(df):
            # 涨停：close >= high_limit * 0.999；跌停：close <= low_limit * 1.001
            limit_up = df["close"] >= df["high_limit"] * 0.999
            limit_dn = df["close"] <= df["low_limit"] * 1.001
            normal = ~(limit_up | limit_dn)
            cu = df.loc[limit_up, "up_down_limit_status"].value_counts().to_dict()
            cd = df.loc[limit_dn, "up_down_limit_status"].value_counts().to_dict()
            cn = df.loc[normal, "up_down_limit_status"].value_counts().to_dict()
            logger.info(f"  涨停日 status 分布: {cu}")
            logger.info(f"  跌停日 status 分布: {cd}")
            logger.info(f"  正常日 status 分布: {cn}")
            result.update(
                {
                    "limit_up_status": {str(k): int(v) for k, v in cu.items()},
                    "limit_dn_status": {str(k): int(v) for k, v in cd.items()},
                    "normal_status": {str(k): int(v) for k, v in cn.items()},
                }
            )
    return result


def probe_bucket_stats(cfg: LiquidityConfig) -> pd.DataFrame:
    """运行全窗口月末 PIT 分档，落盘，并报告每档样本量 / ST 占比统计。"""
    provider = QlibProvider(cfg.pool, cfg.window_start, DATA_END)
    strat_df = stratify(provider, cfg)
    out = DATA_DIR / "strat_membership.parquet"
    strat_df.to_parquet(out, index=False)
    logger.info(f"分档成员表落盘：{out}（{len(strat_df)} 行）")

    # 结构性统计：月末维度
    monthly = strat_df.copy()
    monthly["ym"] = monthly["date"].dt.to_period("M").astype(str)
    # 每月末每轨每档的成员数
    size_pivot = (
        monthly.groupby(["date", "st_track", "bucket"])["code"]
        .count()
        .reset_index()
    )
    logger.info("每档每轨月末成员数（describe）：")
    for (track, bucket), g in size_pivot.groupby(["st_track", "bucket"]):
        logger.info(
            f"  [{track},{bucket}] mean={g['code'].mean():.0f} "
            f"min={g['code'].min()} max={g['code'].max()} "
            f"median={g['code'].median():.0f}"
        )
    return strat_df


def main() -> None:
    ensure_dirs()
    cfg = LiquidityConfig.load()
    logger.info(f"配置：pool={cfg.pool} L={cfg.lookback} H={cfg.predict_len} "
                f"N={cfg.sample_count} window={cfg.window_start}..{cfg.window_end}")

    probe_connection()
    sw_l1_ok = probe_industry_sw_l1(cfg)
    limit_semantics = probe_up_down_limit_status(cfg)
    strat_df = probe_bucket_stats(cfg)

    # 汇总落盘 JSON（结构化阶段 0 结论）
    import json

    summary = {
        "sw_l1_available": sw_l1_ok,
        "limit_status": limit_semantics,
        "n_month_ends": int(strat_df["date"].nunique()),
    }
    out = DATA_DIR / "stage0_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"阶段 0 汇总落盘：{out}")
    logger.info("阶段 0 探测完成。")


if __name__ == "__main__":
    main()
