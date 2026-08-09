# Kronos 接入 qlib 数据层 · 改造计划

> 日期：2026-08-09
> 分支：`feature/qlib-data-layer`（从 `master` 切出）
> 背景：[截面路径zero-shot测试计划_20260809.md](./截面路径zero-shot测试计划_20260809.md) 的阶段 1（akshare 拉取 → parquet 缓存）
> **作废**——本机 qlib（`quant-qlib`，DolphinDB 后端）可直连数据库取数，无需落 CSV 再读。
> 本计划先落地数据层，截面实验待其验收后恢复。

## 0. 已实测的事实（zcode 无需重新探索，直接采信）

全部由本机实跑确认，非推断：

| 项 | 实测值 |
|---|---|
| 解释器 | `/home/user/miniconda3/envs/quant/bin/python`（torch 2.13.0+cu130 / CUDA RTX 5090） |
| qlib | `pyqlib 0.9.6.99`，**editable 安装**指向 `/home/user/workspace/quant-qlib`，任意 cwd 均可 `import qlib` |
| 驱动 | `dolphindb 3.0.4.2` 已装 |
| 连接串 | `DOLPHINDB_URI` 在 `/home/user/workspace/AlphaFarmer/.env`（`dolphindb://admin:***@114.**.**.170:28848`） |
| `factor_core`（AlphaFarmer 的 QlibProvider） | **不可 import**（未安装为包）→ 本项目自建轻量 provider，不得依赖 AlphaFarmer |
| 日频字段 | `open/high/low/close/preclose/volume/amount/factor/vwap/tradestatuscode/limit/stopping` |
| 复权口径 | OHLC 为**后复权**（600000.SH 实测 ~198 元 vs 实际股价 ~13 元） |
| 复权自洽性 | `\|close/preclose-1 − close/Ref(close,1)-1\|` 最大 **8.12e-6**（float32 噪声，容差取 1e-4） |
| `tradestatuscode` | −1 正常(72298) / **0 = 停牌**(162，volume 恒 0、收益恒 0，100% 吻合) / 2 = 除权除息日(430) / 4 = 其他(10) |
| 沪深300 取数性能 | `2024-01-01~2026-08-09` × 6 字段 = **189,444 行，2.2 秒，0 NaN** |
| 成分股口径 | `D.instruments("csi300")` 返回**带区间的 point-in-time 成分**（上述区间内共 365 个不同代码）→ **无幸存者偏差**，原计划的该项局限直接消除 |
| 数据覆盖 | 日频数据至 **2026-08-07**；`D.calendar(freq="day")` 却延伸至 **2040-12-31**（含未来占位日，见 §4 陷阱 3） |
| 可用市场 | `csi300` / `csi500` / `ashares` / `cyb`；`STFilter`（`qlib.data.filter`）仅 DDB 后端可用 |

参考实现（**读，但不 import**）：`/home/user/workspace/AlphaFarmer/factor_core/provider.py`——
其 init-once 双检锁、`_load_lock` 全局串行、`mask_uri` 脱敏日志三处模式直接照搬，其踩坑注释值得逐条读完。

## 1. 目标与非目标

**目标**：给 Kronos 一个直连 qlib 的日频数据层，产出可直接喂给
`KronosPredictor.predict_batch`（[kronos.py:562](../model/kronos.py:562)）的窗口数据，替代"落 CSV 再读"。

**非目标**（明确不做，避免范围蔓延）：

- 不改 `finetune_csv/` 任何现有文件——它服务已封存的分类实验，CSV 路径继续可用；
- 不动 `feature/direction-classifier` 分支；
- 不在本计划内跑截面实验（那是下一步，验收后另起）；
- 不做训练用 Dataset。provider 设计成通用取数即可，训练需要时再基于它加，**本期不写**。

## 2. 交付物

```
kronos_qlib/__init__.py
kronos_qlib/provider.py     # QlibProvider：init-once + 取数
kronos_qlib/windows.py      # 构造 Kronos 推理窗口
tests/test_kronos_qlib.py   # 单测（mock）+ 集成测试（skipif 无 DDB）
docs/Kronos接入qlib数据层说明_20260809.md   # 用法与口径
.env.example                # 只含 DOLPHINDB_URI= 占位，真 .env 已被 .gitignore:63 忽略
```

### 2.1 `provider.py`

```python
class QlibProvider:
    @classmethod
    def init_qlib_once(cls) -> None: ...          # 双检锁，读 .env 的 DOLPHINDB_URI
    def __init__(self, instruments: str | list[str], start_date: str, end_date: str) -> None: ...
    def fetch(self, fields, *, filter_pipe=None, freq="day") -> pd.DataFrame: ...
    def trading_days(self, start=None, end=None) -> pd.DatetimeIndex: ...  # 走 D.calendar
```

硬性要求：

1. `DOLPHINDB_URI` 缺失时抛**说明性异常**（提示去 `.env` 配置），**绝不静默回退**到文件后端或 localhost——
   静默回退会让实验在错数据上跑完还看不出来；
2. `load_dotenv` 必须在 `import qlib` **之前**执行（AlphaFarmer 注释里的坑：`QlibConfig` 的
   dataclass 默认值在类定义时求值）；
3. 日志用 `mask_uri` 脱敏，禁止任何形式打印明文口令；
4. 全部 qlib 数据层调用（`load()` 与 `D.calendar()`）共用**同一把** `_load_lock`——
   两者读同一份进程级缓存 `H["c"]`，分开加锁等于没加（AlphaFarmer 踩过，已撤销双锁）。

### 2.2 `windows.py`

核心函数（签名可微调，语义不可变）：

```python
def build_inference_windows(
    provider, rebalance_date, lookback=90, predict_len=10, pool="csi300",
) -> tuple[list[pd.DataFrame], list[pd.Series], list[pd.Series], list[str]]:
    """构造 predict_batch 所需的四元组：df_list / x_timestamp_list / y_timestamp_list / codes。"""
```

语义规定（每条都是正确性要求，不是风格偏好）：

1. **池按 t 时点取**：`D.list_instruments(D.instruments(pool), start_time=t, end_time=t)`，
   逐个调仓日重取，**不要**一次性取全区间成分当固定池（那会重新引入幸存者偏差）；
2. **窗口取 ≤ t 的最后 L 个交易日**，行数不足 L 的股票**整只跳过**该调仓日（新上市/长期停牌），
   并记录跳过数量；
3. **停牌剔除**：窗口内含 `tradestatuscode == 0` 的交易日 → 该股票该调仓日跳过。
   **不做前向填充**——填充出的假 K 线会被 Kronos 当真实形态编码；
4. **列顺序固定** `["open","high","low","close","volume","amount"]`，与
   `KronosPredictor.price_cols + vol_col + amt_vol`（[kronos.py:489](../model/kronos.py:489)）严格一致；
5. **不要自己做归一化**——`predict` 内部按窗口 z-score + clip 5（[kronos.py:544](../model/kronos.py:544)），
   预归一化会导致双重标准化；
6. `y_timestamp` 取 t 之后的 H 个交易日，**必须来自 `D.calendar`**（节假日不可用 `date + n` 推），
   且须落在真实数据覆盖区间内（见 §4 陷阱 3）；
7. `ST` 过滤留成 `filter_pipe` 可选参数，**默认不启用**（csi300 内 ST 极少，启用与否应在实验层决定，
   不在数据层写死）。

## 3. 验收（每条都要贴实际输出）

**单元测试（无需 DDB，mock `fetch`）**：

1. 行数不足 L → 该股票被跳过，且跳过计数正确；
2. 窗口含 `tradestatuscode == 0` → 跳过；
3. 返回的 df 列顺序与 `KronosPredictor` 期望逐字一致；
4. `y_timestamp` 长度 == `predict_len`，且与 `x_timestamp` 无重叠、严格递增；
5. `DOLPHINDB_URI` 缺失 → 抛说明性异常（不是 `KeyError`，不是静默 localhost）。

**集成测试（`@pytest.mark.skipif` 无 DDB 时跳过）**：

6. **复权自洽性**：`|close/preclose-1 − close/Ref(close,1)-1| < 1e-4`（实测最大 8.12e-6）；
7. **point-in-time 成分**：取 2024-06-28 与 2025-12-31 两个调仓日的 csi300 池，
   两者**不应完全相同**（成分调整必然发生）——若相同说明成分区间没生效，是严重缺陷；
8. **端到端冒烟**：取 1 个调仓日 × 5 只股票 → `build_inference_windows` →
   `KronosPredictor.predict_batch(pred_len=10, sample_count=1)` 跑通，
   输出 5 个 df、每个 10 行、close 列为**有限正数且与输入窗口末值同数量级**
   （量级不符即反归一化出错，是最容易静默错的一环）；
9. **确定性**：固定 `torch.manual_seed(42)`，同输入连跑两次预测结果逐位一致。

**门禁**：9 项全过才算完成。第 8 项是本次改造的真正目的，不可用 mock 代替。

## 4. 已知陷阱（踩过的，别再踩）

1. **`$` 前缀**：`QlibDataLoader` 取数用 `"$close"`，返回后列名需去 `$`（AlphaFarmer 用
   `df.columns.str.replace("$","")`）；
2. **`swap_level=True` 时 level 0 是 datetime**（非 instrument）。reshape 一律按 **level 名**
   `unstack("instrument")`，禁止按位置 `unstack(level=0)`——AlphaFarmer 0.4.0 之前写反过，
   会**静默转置矩阵**，不报错；
3. **日历含未来占位日**：`D.calendar(freq="day")` 到 2040-12-31，而真实数据止于 2026-08-07。
   取 `y_timestamp` 与确定最后一个可评估调仓日时，必须以**数据实际末日**为界，
   否则会拿到一批永远取不到真实收益的"未来调仓日"；
4. **qlib 数据层非线程安全**：单线程跑；确需并发时只能靠 §2.1.4 的全局锁串行；
5. **`filter_pipe` 静默丢弃**：instruments 传 **list** 时 `filter_pipe` 只 warning 不生效
   （`loader.py:237-240`）。若启用 STFilter，instruments 必须传 str/dict。

## 5. 纪律

- 声称任一阶段通过前贴实际输出（`superpowers:verification-before-completion`）；
- 严禁把口令写进代码或文档；`.env` 不入库，只提交 `.env.example`；
- 不动 §1 非目标清单里的任何文件；
- 提交遵循 CLAUDE.md（`feat` 代码 / `test` 测试 / `docs` 文档），`Co-Authored-By: Hugo <shen.lan123@gmail.com>`。

## 6. 本改造对截面实验计划的影响（验收后同步修订）

- 阶段 1（akshare 拉取 + 质检）→ 替换为本数据层，工时从数小时降到秒级；
- 幸存者偏差局限**消除**（point-in-time 成分）；
- 停牌处理从"前向填充"改为"跳过"，口径更干净；
- 阶段 2 的计时探针仍需保留——瓶颈从取数转移到 Kronos 自回归推理本身。
