"""kronos_qlib —— 直连 qlib（DolphinDB 后端）的日频数据层。

为 Kronos 提供可直接喂给 ``KronosPredictor.predict_batch`` 的窗口数据，
替代"落 CSV 再读"。详见 ``docs/Kronos接入qlib数据层说明_20260809.md``。

公开接口：
    - :class:`QlibProvider`：init-once + 取数（fetch / trading_days / list_pool_at）
    - :func:`build_inference_windows`：构造 predict_batch 所需的四元组
"""
from kronos_qlib.provider import QlibProvider
from kronos_qlib.windows import REQUIRED_COLS, build_inference_windows

__all__ = ["QlibProvider", "build_inference_windows", "REQUIRED_COLS"]
