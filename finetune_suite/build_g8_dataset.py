"""G8 数据构建（计划 §1/§3.2，20260820 G8+E1 计划）：语料新鲜度唯一变量落盘。

G8 = G1 配方**逐字一致**，仅改两处日期（计划 §1 冻结表）：

    | 项   | G1                        | G8                          |
    | train | 2011-01-01 ~ 2024-12-31  | 2011-01-01 ~ **2025-06-30** |
    | val   | 2025-01-01 ~ 2025-06-30  | **2025-07-01 ~ 2025-12-31** |

（train 实际起点 = DDB 日频地板 2014-01-02，计划表中已按有效起点表述。）

派生改动（非自由度，声明式对齐）：
    - ``dataset_end_time`` 对齐 val 末 2025-12-31（Config 注释第 6 条同款约定；
      G1 为 2025-06-30）——不延后则 H2 数据取不到；
    - 输出目录 ``finetune_suite/data/g8/``（纪律 §4：不覆盖既有 pkl/权重）。

语料池/采样年份/清洗规则逐字不动：ashares PIT 并集（2011~2025 每年
01-01/07-01 采样）、每股 dropna、全序列 ≥ lookback+predict+1 才保留、
真实 amount。复用 :func:`finetune_suite.build_dataset.build_pickles`。

用法::

    /home/user/miniconda3/envs/quant/bin/python -m finetune_suite.build_g8_dataset
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "finetune"))
sys.path.insert(0, str(_REPO_ROOT))

from finetune_suite.build_dataset import build_pickles, sample_pool_universe
from finetune_suite.train_g1 import G1Config


class G8DataConfig(G1Config):
    """G8 数据配置：在 G1（ashares）之上仅改两处日期 + 派生边界/输出目录。"""

    def __init__(self):
        super().__init__()
        # —— 冻结的两处日期（计划 §1 表）——
        self.train_time_range = ["2011-01-01", "2025-06-30"]  # ① train 终点 +6 个月
        self.val_time_range = ["2025-07-01", "2025-12-31"]    # ② 早停窗改 2025H2
        # —— 派生（声明式对齐，非自由度）——
        self.dataset_end_time = "2025-12-31"  # 对齐 val 末（Config 注释第 6 条）
        # —— 输出目录（纪律 §4：G8 产物入 data/g8/，不覆盖 ashares/）——
        self.dataset_path = str(Path(self.dataset_path).parent / "g8")
        # 构建语境的 instrument（与 G1 的 --pool ashares 构建一致；训练侧不消费）
        self.instrument = "ashares"


def main() -> None:
    cfg = G8DataConfig()
    print(f"[G8-data] train={cfg.train_time_range} val={cfg.val_time_range}")
    print(f"[G8-data] dataset_end={cfg.dataset_end_time} → {cfg.dataset_path}")

    from kronos_qlib.provider import QlibProvider

    pool_provider = QlibProvider(cfg.instrument, cfg.dataset_begin_time, cfg.dataset_end_time)
    universe = sample_pool_universe(pool_provider, cfg.instrument)
    fetch_provider = QlibProvider(universe, cfg.dataset_begin_time, cfg.dataset_end_time)
    stats = build_pickles(fetch_provider, cfg, universe)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    with open(Path(cfg.dataset_path) / "build_stats.json", "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
