# WebUI 接入 kronos_qlib 数据层 · 改造计划

> 日期：2026-08-11
> 新分支：`feature/webui-qlib`，从 `feature/baseline-and-oos` 切出（含 `kronos_qlib` 全链）
> 需求（用户 2026-08-11）：`cd webui && python run.py` 原依赖 `data/` 目录的 CSV，
> **改为 kronos_qlib 直连 DolphinDB 取数**；预测输入 **OHLCVA 六列全部传给模型**。
> 计划我做，zcode 实施。

## 0. 现状事实（已读码确认，直接采信）

| 项 | 现状 |
|---|---|
| 入口 | `webui/run.py`（依赖检查 + 起 `app.py`，Flask 端口 7070）；flask 3.1.3 / plotly 6.8.0 已在 quant 环境 |
| 数据层 | [app.py:60-123](../webui/app.py:60)：扫 `data/` 目录列 CSV/feather → `load_data_file()` 读文件、猜时间列 |
| **amount 被剔除** | [app.py:433-436](../webui/app.py:433) 注释明写 "excluding amount"，只传 OHLC+volume → `KronosPredictor.predict` 内部用 `volume × mean(price)` **合成假 amount**（[kronos.py:531](../model/kronos.py:531)）——本次需求的直接修复点 |
| 既有 bug ① | 未来时间戳用 `pd.date_range(freq=固定间隔)` 外推（[app.py:562-577](../webui/app.py:562)），日频会推出周末/节假日 |
| 既有 bug ② | "latest data" 模式取 `df.iloc[:lookback]`（**文件头部**），名不符实（[app.py:468](../webui/app.py:468)） |
| 既有隐患 ③ | [app.py:165-171](../webui/app.py:165) 连续性分析块缩进错乱，`prediction_results` 为空时 `last_pred` 未定义即引用（NameError 风险） |
| 前端 | 单页 `templates/index.html`（1238 行）：文件下拉选择 + lookback/pred_len/采样参数 + 图表 |
| 模型加载 | `/api/load-model` 支持 mini/small/base，device 参数默认 cpu（本机应默认 `cuda:0`） |

## 1. 目标与非目标

**目标**：数据源从"选文件"变为"选股票"——输入代码（如 `600000.SH`）+ 可选锚定日期，
后端经 `kronos_qlib.QlibProvider` 取日频 OHLCVA，六列**全量**喂给 `predictor.predict`。

**非目标**（防蔓延）：

- 不做多股票批量预测、不做回测（那是 `baseline_suite` 的事）；
- 仅日频（DDB 实例只有日频视图）；
- 不改 `kronos_qlib/` 任何文件；不动 `model/`；
- CSV 路径**整体移除**（不做双模式开关——保留会双倍膨胀前后端分支逻辑；
  需要 CSV 演示时 `git checkout master -- webui` 即可）；
- 前端只做必要改动（文件选择区 → 代码输入区），不重做 UI。

## 2. 设计

### 2.1 数据适配层：`webui/data_source.py`（新增，唯一的数据入口）

```python
def fetch_ohlcva(code: str, *, end_date: str | None, n_bars: int) -> pd.DataFrame:
    """经 QlibProvider 取单只股票末 n_bars 个交易日的 OHLCVA。

    :returns: 列 = [timestamps, open, high, low, close, volume, amount]，
        timestamps 为真实交易日；行数可能 < n_bars（新股/长停牌），由调用方判断。
    """

def future_trading_days(after: pd.Timestamp, n: int) -> pd.Series: ...
def validate_code(code: str) -> bool: ...      # 在 ashares 池内校验存在性
def list_pool(pool: str = "csi300") -> list[str]: ...
```

语义规定（正确性要求）：

1. **六列全取全传**：`open/high/low/close/volume/amount` 一列不剔——修复 §0 的 amount 合成问题；
2. 停牌日在 DDB 无行，**不前向填充**——窗口即"末 N 个有数据的交易日"；
   窗口内含 `tradestatuscode != -1` 的行数在响应里回报给前端展示（提示数据含停牌前夕/除权日）；
3. `future_trading_days` 走 `provider.trading_days()`（日历延伸到 2040，取未来交易日**正当**——
   这是预测时间戳，不是评估边界），一并修掉 §0 bug ①：预测/对比时间戳一律来自交易日历，
   不再 `date_range` 外推；
4. provider 每请求新建实例即可（进程内 qlib init-once 已由 `QlibProvider` 保证；
   Flask 需以**单线程/单 worker** 跑，见 §4 陷阱 1）。

### 2.2 API 改造（`app.py`）

| 原 | 新 | 语义 |
|---|---|---|
| `GET /api/data-files` | `GET /api/instruments?pool=csi300` | 返回池内代码列表（前端下拉/搜索用）；另接受任意 `ashares` 内代码直输 |
| `POST /api/load-data {file_path}` | `POST /api/load-data {code, end_date?}` | 返回该股数据概况：可用行数、首末日期、价格区间（标注**后复权**）、频率恒 "1 day" |
| `POST /api/predict {file_path, ...}` | `POST /api/predict {code, anchor_date?, lookback, pred_len, ...}` | 见下 |

`/api/predict` 语义（同时修掉 §0 bug ②）：

- `anchor_date` 为空 → **真·最新**：取数据末日为锚 t，窗口 = ≤t 的末 `lookback` 根，
  预测 `pred_len` 根，无对比段（未来未发生）；
- `anchor_date` 给定 → 历史回看模式：窗口 = ≤anchor 的末 `lookback` 根，
  预测与其后 `pred_len` 根真实数据对比（数据不足 pred_len 时对比段截短并提示）;
- `x_df` 传**六列**；`x_timestamp/y_timestamp` 为 `pd.Series`（保持 [app.py:474-477](../webui/app.py:474) 的兼容处理）；
- 默认参数改为日频合理值：`lookback=400 → 90`、`pred_len=120 → 10`（前端默认值同步，
  上限仍受 `max_context=512` 约束）；
- 顺手修 §0 隐患 ③（该块在保存路径上，属本次触碰范围内的缺陷，非无关重构）。

### 2.3 前端（`templates/index.html`，最小改动）

- 文件下拉区 → 代码输入框（datalist 用 `/api/instruments` 填充，支持手输任意 A股代码）+
  可选锚定日期（date input）；
- 数据概况卡片提示"价格为后复权口径"；
- lookback/pred_len 默认值改 90/10；其余（模型选择、采样参数、图表）不动；
- 模型加载的 device 默认值改 `cuda:0`（下拉保留 cpu 选项）。

## 3. 验收（每条贴实际输出）

**单元测试（`tests/test_webui_data_source.py`，FakeProvider 模式，沿用 `test_kronos_qlib.py` 风格）**：

1. `fetch_ohlcva` 返回**恰好七列**、列序固定、`timestamps` 严格递增；
2. 行数不足 `n_bars` → 如实返回并可判断（不填充、不报错）；
3. `future_trading_days` 跨周末/节假日正确（fake 日历含假期，断言跳过）；
4. `validate_code`：池外代码 → False；
5. **amount 直传断言**：mock `predictor.predict`，断言收到的 df **含 amount 列**且值来自数据层
   （非 volume×均价合成）——这是本需求的核心验收，用注入式验证：把 amount 列改为 NaN 剔除后，
   断言走到合成分支（证明该断言有判别力）。

**集成测试（skipif 无 DDB）**：

6. `fetch_ohlcva("600000.SH", n_bars=90)` → 90 行、无 NaN、价格量级与后复权一致（数百元）；
7. Flask test client 全链路：`/api/load-data` → `/api/predict`（kronos-small + cpu 或
   kronos-base + cuda）→ 返回 `prediction_results` 恰 `pred_len` 条、close 有限正数且与
   窗口末值同数量级、时间戳全为交易日（无周末）；
8. 历史回看模式：`anchor_date=2025-06-16` → 有对比段，长度 = pred_len。

**人工冒烟（zcode 执行并截图/贴 log）**：

9. `cd webui && python run.py` 起服务 → 浏览器加载 kronos-base（cuda:0）→
   输入 `600000.SH` 预测 → 图表渲染正常、K 线连续无周末空洞。

## 4. 已知陷阱

1. **qlib 数据层非线程安全**：Flask 默认多线程（`app.run(threaded=True)` 是默认值）。
   必须 `app.run(..., threaded=False)` 单线程跑，或全部数据访问过 `QlibProvider._load_lock`
   （provider 已内置锁，但 webui 层不要绕过 provider 直调 `D.*`）；`debug=True` 的
   reloader 会双进程 init qlib，改 `debug=False` 或 `use_reloader=False`；
2. **`.env` 缺失时**：`QlibProvider` 抛说明性 RuntimeError——webui 启动时预检并在页面/日志
   给出可读提示，不许静默降级回 CSV（CSV 路径已删）；
3. **后复权价与现价差异大**（600000.SH 约 198 vs 13 元），前端必须标注口径，
   否则用户会以为数据错了；
4. **日历 2040 占位日**：仅用于未来时间戳属正当用法；但历史回看模式的对比段边界
   必须以数据实际末日截断（沿用既有认知，勿再踩）；
5. `predictor.predict` 单股串行即可，勿引入并发（陷阱 1）。

## 5. 纪律

- 每阶段贴实际输出（`superpowers:verification-before-completion`）；验收 9 项全过才算完成，
  第 9 项必须真起服务，不可用 test client 代替；
- 不改 `kronos_qlib/`、`model/`、四个实验目录；`webui/prediction_results/` 产物不入库；
- 提交遵循 CLAUDE.md（首个提交 = 本计划），`Co-Authored-By: Hugo <shen.lan123@gmail.com>`；
  完成后推送分支。
