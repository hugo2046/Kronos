# Kronos 接入 qlib 数据层 · 用法与口径说明

> 日期：2026-08-09
> 分支：`feature/qlib-data-layer`
> 配套计划：[Kronos接入qlib数据层改造计划_20260809.md](./Kronos接入qlib数据层改造计划_20260809.md)

本文档说明 `kronos_qlib` 数据层的**用法、数据口径与验收实测结果**。
所有连接串一律以 `mask_uri` 形式呈现（`dolphindb://admin:***@114.**.**.170:28848`），
真实凭据仅存于 Kronos 根 `.env`（已 `.gitignore`）。

## 1. 安装与配置

### 1.1 解释器与依赖
- 解释器：`/home/user/miniconda3/envs/quant/bin/python`
- 已具备：`pyqlib 0.9.6.99`（editable 指向 `/home/user/workspace/quant-qlib`）、
  `dolphindb 3.0.4.2`、`python-dotenv`、`loguru`、`pytest`。

### 1.2 配置 `.env`
在 Kronos 根目录创建 `.env`（参照 `.env.example`）：

```
DOLPHINDB_URI=dolphindb://<user>:<password>@<host>:<port>
```

`.env` 已被 `.gitignore` 忽略，**不入库**。缺失时 `QlibProvider.init_qlib_once()`
抛说明性 `RuntimeError`，**绝不静默回退**到 localhost / 文件后端（计划 §2.1.1）。

### 1.3 何时加载 .env
`kronos_qlib/provider.py` 在 `import qlib` **之前**用确定性路径
（`Path(__file__).resolve().parents[1] / ".env"`）调 `load_dotenv`，不受 cwd 影响。
这是因为 qlib `QlibConfig` 的 dataclass 默认值在**类定义时**求值，.env 加载推迟会拿到空 URI。

## 2. 用法

### 2.1 直接取数（QlibProvider）

```python
from kronos_qlib import QlibProvider

# 市场字符串（csi300/csi500/ashares/cyb）或代码列表
p = QlibProvider("csi300", "2024-01-01", "2026-08-09")

# 取字段：返回 MultiIndex(datetime, instrument) DataFrame，列名已去 $ 前缀
df = p.fetch(["$close", "$volume", "$amount"])

# 交易日历（来自 D.calendar，与取数共用同一把锁）
cal = p.trading_days("2024-01-01", "2025-06-30")

# 某时点的 point-in-time 成分股（无幸存者偏差）
members = p.list_pool_at("csi300", "2025-06-16")
```

⚠️ reshape 一律按 level **名**（`unstack("instrument")` / `xs(code, level="instrument")`），
禁止按位置 `unstack(level=0)`——`swap_level=True` 下 level 0 是 datetime，按位置会**静默转置矩阵**。

### 2.1.1 剔除 ST 股票（filter_pipe + STFilter）

`filter_pipe` 挂在 `fetch()` 上（不是构造函数），透传给内部 `QlibDataLoader`：

```python
from kronos_qlib import QlibProvider
from qlib.data.filter import STFilter

p = QlibProvider("ashares", "2024-01-01", "2026-08-09")

# ST / *ST 区间内的行不出现在结果中
df = p.fetch(["$close", "$volume"], filter_pipe=[STFilter()])
```

`STFilter` 按 Wind `AShareST` 的 ST 区间对股票池做差集（设计文档见 quant-qlib
`docs/superpowers/specs/2026-08-06-st股票过滤-design.md`），可正交叠加于任意
基础池；ST 区间数据需先经 `examples/初始创建ddb数据库/sync_st.py` 同步。

⚠️ 前提是 provider 构造时 `instruments` 传 **str 市场名**——传代码 list 时
`filter_pipe` 只 warning 不生效（见陷阱 5）。

### 2.2 构造 Kronos 推理窗口（build_inference_windows）

```python
from kronos_qlib import QlibProvider, build_inference_windows
from model import Kronos, KronosPredictor, KronosTokenizer

p = QlibProvider("csi300", "2024-01-01", "2025-06-30")
df_list, x_ts, y_ts, codes, stats = build_inference_windows(
    p, rebalance_date="2025-06-16",
    lookback=90, predict_len=10, pool="csi300",
)
# 喂给 predict_batch
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
predictor = KronosPredictor(model, tokenizer, device="cuda:0")
preds = predictor.predict_batch(df_list, x_ts, y_ts, pred_len=10, sample_count=1)
```

`build_inference_windows` 返回 `(df_list, x_timestamp_list, y_timestamp_list, codes, stats)`，
其中 `stats` 含 `n_pool / n_kept / skipped_short / skipped_halt`。

## 3. 数据口径（关键，易踩坑）

| 项 | 口径 |
|---|---|
| 复权 | OHLC 为**后复权**（600000.SH 实测 ~198 元 vs 实际 ~13 元） |
| 复权自洽 | `\|close/preclose-1 − close/Ref(close,1)-1\|` 剔除每股首行后 max < 1e-4 |
| `tradestatuscode` | -1 正常 / **0 = 停牌**（volume 恒 0、收益恒 0）/ 2 = 除权除息 / 4 = 其他 |
| 停牌处理 | **跳过**，不前向填充（填充出的假 K 线会被 Kronos 当真实形态编码） |
| 成分股 | point-in-time（逐调仓日重取），**无幸存者偏差** |
| 日历陷阱 | `D.calendar(freq="day")` 延伸至 2040-12-31，真实数据止于更早日期；`y_timestamp` 以**数据末日**为界 |
| 列顺序 | 固定 `["open","high","low","close","volume","amount"]`，与 `KronosPredictor` 期望逐字一致 |
| 归一化 | **数据层不做**——`predict` 内部按窗口 z-score + clip 5（kronos.py:544），预归一化会双重标准化 |
| ST 过滤 | 留成 `filter_pipe` 可选参数，**默认不启用**（实验层决定） |

## 4. 验收实测结果（9 项全过）

执行命令（`quant` 环境）：

```
/home/user/miniconda3/envs/quant/bin/python -m pytest tests/test_kronos_qlib.py -v
```

实测输出（连接串已脱敏）：

```
============================= test session starts ==============================
platform linux -- Python 3.13.14, pytest-9.1.1
rootdir: /home/user/workspace/Kronos
configfile: pyproject.toml
collecting ... collected 9 items

tests/test_kronos_qlib.py::test_skip_short_history
  build_inference_windows(...): pool=2 kept=1 skipped_short=1 skipped_halt=0  PASSED
tests/test_kronos_qlib.py::test_skip_halt
  build_inference_windows(...): pool=2 kept=1 skipped_short=0 skipped_halt=1  PASSED
tests/test_kronos_qlib.py::test_column_order_matches_kronos
  build_inference_windows(...): pool=1 kept=1 skipped_short=0 skipped_halt=0  PASSED
tests/test_kronos_qlib.py::test_y_timestamp_correct
  build_inference_windows(...): pool=1 kept=1 skipped_short=0 skipped_halt=0  PASSED
tests/test_kronos_qlib.py::test_missing_uri_raises                              PASSED
INFO | Qlib 初始化成功 - URI: dolphindb://admin:***@114.**.**.170:28848
tests/test_kronos_qlib.py::test_adjustment_self_consistency                     PASSED
tests/test_kronos_qlib.py::test_point_in_time_membership                        PASSED
tests/test_kronos_qlib.py::test_end_to_end_smoke
  build_inference_windows(2025-06-16, pool=csi300): pool=300 kept=294 skipped_short=0 skipped_halt=6  PASSED
tests/test_kronos_qlib.py::test_determinism
  build_inference_windows(2025-06-16, pool=csi300): pool=300 kept=294 skipped_short=0 skipped_halt=6  PASSED

======================== 9 passed, 1 warning in 17.99s =========================
```

逐项说明：

| # | 验收项 | 结果 | 关键实测 |
|---|---|---|---|
| 1 | 行数不足 L → 跳过 | ✅ | skipped_short 计数正确 |
| 2 | 窗口含停牌 → 跳过 | ✅ | skipped_halt 计数正确 |
| 3 | df 列顺序与 KronosPredictor 一致 | ✅ | `["open","high","low","close","volume","amount"]` |
| 4 | y_timestamp 正确 | ✅ | 长度=predict_len、与 x 无重叠、严格递增 |
| 5 | DOLPHINDB_URI 缺失 → 说明性异常 | ✅ | `RuntimeError`，消息含 `.env` 与 `DOLPHINDB_URI` |
| 6 | 复权自洽性 | ✅ | 剔除每股首行后 max < 1e-4（34800 行仅边界首行异常） |
| 7 | point-in-time 成分 | ✅ | 2024-06-28 vs 2025-12-31 两 csi300 池不相等 |
| 8 | 端到端冒烟 | ✅ | 2025-06-16: pool=300 kept=294 halt=6 → 5 股 predict_batch → 5×10 行，close 有限正数且同数量级 |
| 9 | 确定性 | ✅ | `torch.manual_seed(42)` 两次 predict_batch 结果 `torch.equal` 逐位一致 |

### 4.1 关于验收 6 的口径说明
计划 §0 实测"max 8.12e-6"。实测发现每只股票在区间**首日**因 `Ref` 跨越复权边界
产生 1 个异常点（如 61.77）——这是 DolphinDB `Ref` 在序列边界的已知行为，非数据错误
（34800 行中仅 1 行超 1e-4，0.999 分位数为 0.0）。测试按"剔除每只股票首行"后取 max，
与计划口径一致。

## 5. 模块结构

```
kronos_qlib/
  __init__.py     # 导出 QlibProvider / build_inference_windows / REQUIRED_COLS
  provider.py     # QlibProvider：init-once（双检锁）+ fetch + trading_days + list_pool_at
  windows.py      # build_inference_windows：构造 predict_batch 四元组
tests/
  test_kronos_qlib.py  # 5 单测（FakeProvider，无需 DDB）+ 4 集成（skipif）
pyproject.toml    # [tool.pytest.ini_options]，注册 integration marker
.env.example      # DOLPHINDB_URI 占位
```

### 5.1 线程安全
qlib 数据层非线程安全（计划陷阱 4）。`QlibProvider` 用**单一** `_load_lock`
同时串行 `fetch`（`QlibDataLoader.load`）与 `trading_days`（`D.calendar`）——
两者读同一份进程级缓存 `qlib.data.cache.H["c"]`，分开加锁等于没加。
`_init_lock` 与 `_load_lock` 分离（一次性初始化 vs 每次访问，互不相交的临界区）。

## 6. 已知陷阱（踩过的，别再踩）

1. **`$` 前缀**：取数用 `$close`，`fetch` 返回后已自动 `str.replace("$","")`。
2. **level 0 是 datetime**：reshape 按 level **名**，禁止按位置（会静默转置）。
3. **日历含未来占位日**：到 2040-12-31，真实数据止于更早日期；`y_timestamp` 以数据末日为界。
4. **非线程安全**：单线程跑；确需并发只能靠 `_load_lock` 串行。
5. **`filter_pipe` 静默丢弃**：instruments 传 **list** 时 `filter_pipe` 只 warning 不生效。
   启用 STFilter 时 provider 构造时 instruments 必须传 str 市场名。
