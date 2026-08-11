"""Kronos Web UI 后端（kronos_qlib 数据层版本）。

数据源从"选文件读 CSV"改为"选股票经 :mod:`webui.data_source` 直连 DolphinDB"，
OHLCVA 六列全量传给 :class:`model.kronos.KronosPredictor`。详见
``docs/webui接入qlib数据层计划_20260811.md``。

CSV 路径已整体移除（计划 §1 非目标——不做双模式开关）。需要 CSV 演示时
``git checkout master -- webui``。
"""
import os
import datetime
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

import sys
import warnings

warnings.filterwarnings("ignore")

# Add project root directory to path（model / kronos_qlib 在仓库根）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import Kronos, KronosPredictor, KronosTokenizer

    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Kronos model cannot be imported, will use simulated data for demonstration")

# 数据层：webui 唯一的数据入口（经 kronos_qlib 取日频 OHLCVA）
from webui import data_source
from webui.data_source import OHLCVA_COLS, OUTPUT_COLS

# 启动预检 DOLPHINDB_URI（计划陷阱 2）：缺失给可读提示，绝不静默降级回 CSV。
_ENV_ERROR = data_source.check_env()
if _ENV_ERROR:
    print("⚠️  " + _ENV_ERROR)

app = Flask(__name__)
CORS(app)

# Global variables to store models
tokenizer = None
model = None
predictor = None

# Available model configurations
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}

# 日频合理默认值（计划 §2.2）：lookback 400→90、pred_len 120→10，
# 上限仍受 max_context=512 约束（前端默认值同步）。
DEFAULT_LOOKBACK = 90
DEFAULT_PRED_LEN = 10
MAX_CONTEXT = 512


def save_prediction_results(code, prediction_type, prediction_results, actual_data,
                            input_data, prediction_params):
    """Save prediction results to file（产物不入库，仅本地留痕）。"""
    try:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)

        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'code': code,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }

        # 连续性分析：仅当预测段与实际段都非空时才计算（修计划隐患 ③——
        # 旧代码 prediction_results 为空时 last_pred 未定义即引用，NameError 风险）。
        if actual_data and len(actual_data) > 0 and prediction_results and len(prediction_results) > 0:
            last_pred = prediction_results[0]   # 第一个预测点
            first_actual = actual_data[0]       # 第一个实际点

            save_data['analysis']['continuity'] = {
                'last_prediction': {
                    'open': last_pred['open'], 'high': last_pred['high'],
                    'low': last_pred['low'], 'close': last_pred['close']
                },
                'first_actual': {
                    'open': first_actual['open'], 'high': first_actual['high'],
                    'low': first_actual['low'], 'close': first_actual['close']
                },
                'gaps': {
                    'open_gap': abs(last_pred['open'] - first_actual['open']),
                    'high_gap': abs(last_pred['high'] - first_actual['high']),
                    'low_gap': abs(last_pred['low'] - first_actual['low']),
                    'close_gap': abs(last_pred['close'] - first_actual['close']),
                },
                'gap_percentages': {
                    'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                    'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                    'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                    'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100,
                }
            }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"Prediction results saved to: {filepath}")
        return filepath

    except Exception as e:
        print(f"Failed to save prediction results: {e}")
        return None


def create_prediction_chart(historical_df, pred_df, y_timestamp, actual_df=None):
    """创建预测图表。

    时间戳一律来自真实交易日（historical_df['timestamps'] / y_timestamp /
    actual_df['timestamps']），修掉旧版用 ``pd.date_range(freq=固定间隔)``
    外推会推出周末/节假日的 bug（计划 §0 bug ①）。

    :param historical_df: 历史窗口，含 ``timestamps`` 列 + OHLCVA。
    :param pred_df: 预测结果，index 对齐 ``y_timestamp``。
    :param y_timestamp: 预测段交易日 ``pd.Series``（长度 == pred_len）。
    :param actual_df: 可选对比段，含 ``timestamps`` 列 + OHLCVA。
    """
    # ⚠️ 一律用 Python list 构造 trace，不要直接传 Series/ndarray：
    # plotly.py ≥5 会把 **numpy 数组**序列化成 base64 二进制
    # （``{"dtype": "f4", "bdata": "..."}``），**只有 plotly.js v3 能解析**；
    # 前端若是旧版（如被官方冻结在 v1.58.5 的 plotly-latest 别名），candlestick
    # 取不到 open/high/low/close，图表**静默空白且无报错**。传 list 则输出普通
    # JSON 数组，对前端版本不敏感。由 test_chart_json_uses_plain_arrays 锁定。
    def _ohlc(frame):
        """取 OHLC 四列并转为 Python list（避免 base64 二进制序列化）。"""
        return {k: frame[k].astype(float).tolist() for k in ('open', 'high', 'low', 'close')}

    def _x(values):
        """时间戳转 ISO 字符串 list（category 轴按标签渲染）。"""
        return [pd.Timestamp(v).isoformat() for v in pd.Series(values)]

    fig = go.Figure()

    # 历史数据（K 线）
    fig.add_trace(go.Candlestick(
        x=_x(historical_df['timestamps']), **_ohlc(historical_df),
        name=f'Historical ({len(historical_df)} bars)',
        increasing_line_color='#26A69A', decreasing_line_color='#EF5350'
    ))

    # 预测数据（K 线，时间戳来自交易日历）
    if pred_df is not None and len(pred_df) > 0:
        fig.add_trace(go.Candlestick(
            x=_x(y_timestamp), **_ohlc(pred_df),
            name=f'Prediction ({len(pred_df)} bars)',
            increasing_line_color='#66BB6A', decreasing_line_color='#FF7043'
        ))

    # 对比实际数据（K 线）
    if actual_df is not None and len(actual_df) > 0:
        fig.add_trace(go.Candlestick(
            x=_x(actual_df['timestamps']), **_ohlc(actual_df),
            name=f'Actual ({len(actual_df)} bars)',
            increasing_line_color='#FF9800', decreasing_line_color='#F44336'
        ))

    fig.update_layout(
        title='Kronos Prediction (daily, 后复权)',
        xaxis_title='Trading Day', yaxis_title='Price (后复权)',
        template='plotly_white', height=600, showlegend=True,
        # x 轴用 category 而非 date：plotly date 轴会把周末 / 节假日画成空白列，
        # category 轴让连续交易日相邻无间隙——做到"K 线连续无周末空洞"。
        xaxis={'type': 'category', 'rangeslider': {'visible': False}},
    )
    # 数组已在上方转为 Python list，此处输出即为纯 JSON 数组。
    # 前端 CDN 亦锁定 plotly.js v3（见 templates/index.html），二者双保险。
    return fig.to_json(engine="json")


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/instruments')
def get_instruments():
    """返回池内代码列表（前端下拉 / 搜索用），默认 csi300。

    另接受任意 ashares 内代码直输（前端 datalist 仅作提示，不强制选择）。
    """
    pool = request.args.get('pool', 'csi300')
    try:
        members = data_source.list_pool(pool)
        return jsonify({'pool': pool, 'instruments': members, 'count': len(members)})
    except Exception as e:
        return jsonify({'error': f'Failed to list pool {pool}: {str(e)}'}), 500


@app.route('/api/load-data', methods=['POST'])
def load_data():
    """加载单只股票数据概况。

    入参 ``{code, end_date?}``；返回可用行数、首末日期、价格区间（标注**后复权**）、
    频率恒 ``"1 day"``、非交易态行数。
    """
    if _ENV_ERROR:
        return jsonify({'error': _ENV_ERROR}), 500
    try:
        data = request.get_json() or {}
        code = (data.get('code') or '').strip()
        end_date = (data.get('end_date') or '').strip() or None

        if not code:
            return jsonify({'error': '股票代码不能为空（如 600000.SH）'}), 400

        # 概览取满可用上限（MAX_CONTEXT），让用户知道 lookback 上界。
        df = data_source.fetch_ohlcva(code, end_date=end_date, n_bars=MAX_CONTEXT)
        if len(df) == 0:
            return jsonify({'error': f'{code} 在近区间无任何数据，请确认代码或换只股票'}), 400

        price_cols = ['open', 'high', 'low', 'close']
        data_info = {
            'code': code,
            'rows': len(df),
            'columns': OUTPUT_COLS,
            'start_date': df['timestamps'].iloc[0].isoformat(),
            'end_date': df['timestamps'].iloc[-1].isoformat(),
            'price_range': {
                'min': float(df[price_cols].min().min()),
                'max': float(df[price_cols].max().max()),
            },
            # 标注后复权口径（陷阱 3）：600000.SH 约 198 vs 现价 13 元，否则用户以为数据错了
            'adjustment': '后复权（post-adjusted）',
            'frequency': '1 day',
            'prediction_columns': OHLCVA_COLS,
            'non_tradeable_rows': int(df.attrs.get('non_tradeable_rows', 0)),
        }

        return jsonify({
            'success': True,
            'data_info': data_info,
            'message': f'成功加载 {code}，共 {len(df)} 个交易日（后复权）'
        })

    except Exception as e:
        return jsonify({'error': f'加载数据失败：{str(e)}'}), 500


def run_prediction(df_window, predictor_obj, x_timestamp, y_timestamp, *,
                   pred_len, T, top_p, sample_count):
    """从窗口构造六列输入并调用 ``predictor.predict``。

    抽成函数便于单测注入 mock predictor：断言收到的 df **含 amount 列**且值
    来自数据层（计划 §3 验收 5）。
    """
    # 六列全量直传（计划核心修复点）：不剔 amount，避免 kronos.py:531 合成假 amount。
    x_df = df_window[OHLCVA_COLS]
    pred_df = predictor_obj.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=T,
        top_p=top_p,
        sample_count=sample_count,
    )
    return pred_df


@app.route('/api/predict', methods=['POST'])
def predict():
    """执行预测。

    入参 ``{code, anchor_date?, lookback, pred_len, temperature, top_p, sample_count}``。

    - ``anchor_date`` 为空 → 真·最新模式：窗口 = 末 ``lookback`` 根（**文件/数据尾部**，
      修计划 §0 bug ②——旧版 iloc[:lookback] 取的是头部），预测 ``pred_len`` 根，
      无对比段（未来未发生）；
    - ``anchor_date`` 给定 → 历史回看模式：窗口 = ≤anchor 末 ``lookback`` 根，
      预测与其后 ``pred_len`` 根真实数据对比（不足 pred_len 时截短并提示）。

    预测 / 对比时间戳一律来自交易日历（修 bug ①）。
    """
    if _ENV_ERROR:
        return jsonify({'error': _ENV_ERROR}), 500
    if not (MODEL_AVAILABLE and predictor is not None):
        return jsonify({'error': 'Kronos 模型未加载，请先加载模型'}), 400

    try:
        data = request.get_json() or {}
        code = (data.get('code') or '').strip()
        anchor_date = (data.get('anchor_date') or '').strip() or None
        lookback = int(data.get('lookback', DEFAULT_LOOKBACK))
        pred_len = int(data.get('pred_len', DEFAULT_PRED_LEN))
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))

        if not code:
            return jsonify({'error': '股票代码不能为空（如 600000.SH）'}), 400
        if lookback <= 0 or pred_len <= 0:
            return jsonify({'error': 'lookback / pred_len 必须为正整数'}), 400
        if lookback > MAX_CONTEXT:
            return jsonify({'error': f'lookback 超过模型上下文上限 {MAX_CONTEXT}'}), 400

        historical_mode = anchor_date is not None

        # —— 取数：模式 A 取末 lookback；模式 B 取 anchor 前后覆盖 lookback+pred_len ——
        if historical_mode:
            anchor_ts = pd.Timestamp(anchor_date)
            # 末日放宽到 anchor 之后，确保拿到 pred_len 个真实对比交易日。
            fetch_end = (anchor_ts + pd.Timedelta(days=pred_len * 2 + 30)).strftime('%Y-%m-%d')
            fetch_n = lookback + pred_len + 30
            full = data_source.fetch_ohlcva(code, end_date=fetch_end, n_bars=fetch_n)
            if len(full) == 0:
                return jsonify({'error': f'{code} 在该区间无数据'}), 400

            window = full[full['timestamps'] <= anchor_ts].iloc[-lookback:]
            # 对比段：anchor 之后真实数据的头 pred_len 根（不足则截短）
            compare_df = full[full['timestamps'] > anchor_ts].iloc[:pred_len]
        else:
            full = data_source.fetch_ohlcva(code, end_date=None, n_bars=lookback)
            if len(full) == 0:
                return jsonify({'error': f'{code} 在近区间无数据'}), 400
            window = full
            compare_df = None

        if len(window) < lookback:
            return jsonify({
                'error': f'可用历史不足：需要 {lookback} 个交易日，'
                         f'{code} 仅有 {len(window)} 个（新股 / 长停牌）。可减小 lookback 重试。'
            }), 400

        # —— 时间戳来自交易日历（修 bug ①）——
        x_timestamp = window['timestamps'].reset_index(drop=True)
        anchor_eff = pd.Timestamp(anchor_date) if historical_mode else window['timestamps'].iloc[-1]
        y_timestamp = data_source.future_trading_days(anchor_eff, pred_len)
        if len(y_timestamp) < pred_len:
            return jsonify({'error': f'锚定日之后交易日不足 {pred_len} 个（日历末端）'}), 400

        prediction_type = (
            f"历史回看：锚定 {anchor_eff.strftime('%Y-%m-%d')}，"
            f"窗口末 {lookback} 根 → 预测 {pred_len} 根，对比真实 {len(compare_df) if compare_df is not None else 0} 根"
            if historical_mode else
            f"真·最新：锚定数据末日 {anchor_eff.strftime('%Y-%m-%d')}，"
            f"窗口末 {lookback} 根 → 预测 {pred_len} 根（未来未发生，无对比段）"
        )

        # —— 调用模型（六列直传）——
        try:
            pred_df = run_prediction(
                window, predictor, x_timestamp, y_timestamp,
                pred_len=pred_len, T=temperature, top_p=top_p, sample_count=sample_count,
            )
        except Exception as e:
            return jsonify({'error': f'Kronos 模型预测失败：{str(e)}'}), 500

        # —— 对比段（仅历史回看）——
        actual_data = []
        actual_chart_df = None
        if compare_df is not None and len(compare_df) > 0:
            actual_chart_df = compare_df
            for _, row in compare_df.iterrows():
                actual_data.append({
                    'timestamp': row['timestamps'].isoformat(),
                    'open': float(row['open']), 'high': float(row['high']),
                    'low': float(row['low']), 'close': float(row['close']),
                    'volume': float(row['volume']), 'amount': float(row['amount']),
                })

        # —— 预测结果序列（时间戳 = 交易日历）——
        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            prediction_results.append({
                'timestamp': y_timestamp.iloc[i].isoformat() if i < len(y_timestamp) else f"T{i}",
                'open': float(row['open']), 'high': float(row['high']),
                'low': float(row['low']), 'close': float(row['close']),
                'volume': float(row['volume']), 'amount': float(row['amount']),
            })

        chart_json = create_prediction_chart(window, pred_df, y_timestamp, actual_chart_df)

        # 本地留痕（产物不入库）
        try:
            save_prediction_results(
                code=code, prediction_type=prediction_type,
                prediction_results=prediction_results, actual_data=actual_data,
                input_data=window,
                prediction_params={
                    'lookback': lookback, 'pred_len': pred_len,
                    'temperature': temperature, 'top_p': top_p,
                    'sample_count': sample_count,
                    'anchor_date': anchor_date or 'latest',
                },
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")

        msg = f'预测完成，生成 {pred_len} 个预测点'
        if len(actual_data) > 0:
            note = '' if len(actual_data) >= pred_len else f'（对比段仅 {len(actual_data)} 根，数据不足已截短）'
            msg += f'，含 {len(actual_data)} 个真实对比点{note}'

        return jsonify({
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'has_comparison': len(actual_data) > 0,
            'message': msg,
        })

    except Exception as e:
        return jsonify({'error': f'预测失败：{str(e)}'}), 500


@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load Kronos model"""
    global tokenizer, model, predictor

    try:
        if not MODEL_AVAILABLE:
            return jsonify({'error': 'Kronos model library not available'}), 400

        data = request.get_json() or {}
        model_key = data.get('model_key', 'kronos-small')
        # 默认 cuda:0（计划 §2.3）；前端下拉保留 cpu 选项
        device = data.get('device', 'cuda:0')

        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': f'Unsupported model: {model_key}'}), 400

        model_config = AVAILABLE_MODELS[model_key]

        tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_id'])
        model = Kronos.from_pretrained(model_config['model_id'])
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config['context_length'])

        return jsonify({
            'success': True,
            'message': f'Model loaded successfully: {model_config["name"]} ({model_config["params"]}) on {device}',
            'model_info': {
                'name': model_config['name'],
                'params': model_config['params'],
                'context_length': model_config['context_length'],
                'description': model_config['description']
            }
        })

    except Exception as e:
        return jsonify({'error': f'Model loading failed: {str(e)}'}), 500


@app.route('/api/available-models')
def get_available_models():
    """Get available model list"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'model_available': MODEL_AVAILABLE
    })


@app.route('/api/model-status')
def get_model_status():
    """Get model status"""
    if MODEL_AVAILABLE:
        if predictor is not None:
            return jsonify({
                'available': True,
                'loaded': True,
                'message': 'Kronos model loaded and available',
                'current_model': {
                    'name': predictor.model.__class__.__name__,
                    'device': str(next(predictor.model.parameters()).device)
                }
            })
        else:
            return jsonify({
                'available': True,
                'loaded': False,
                'message': 'Kronos model available but not loaded'
            })
    else:
        return jsonify({
            'available': False,
            'loaded': False,
            'message': 'Kronos model library not available, please install related dependencies'
        })


if __name__ == '__main__':
    print("Starting Kronos Web UI (kronos_qlib data layer)...")
    print(f"Model availability: {MODEL_AVAILABLE}")
    if _ENV_ERROR:
        print("⚠️  " + _ENV_ERROR)
    # 计划陷阱 1：qlib 数据层非线程安全——必须单线程跑；debug reloader 会双进程
    # init qlib，故 debug=False。
    app.run(debug=False, host='0.0.0.0', port=7070, threaded=False)
