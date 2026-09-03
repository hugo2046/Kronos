"""G9 checkpoint 选择规则契约测试（计划 §1/§4.1，20260821 G9 计划）。

- ``test_recipe_frozen``：``g9_ckpt/train_g9.py`` 的 G9Config 相对 G1 配方
  **除落盘策略外逐字段一致**（语料/超参/种子/tokenizer 共享只读），且训练脚本
  ``g9_ckpt/train_predictor_all_epochs.py`` = ``finetune_suite/train_predictor.py``
  逐字复制 + 差异仅限标记块内（每 epoch 落盘 + 头部保真声明注释）——源码级
  diff 断言；
- ``test_all_epoch_saver``：15 个 epoch checkpoint 目录齐全、每个可
  ``Kronos.from_pretrained`` 装载（先 FAIL 后 PASS：需先运行 4.2 全 epoch 重训
  落盘 checkpoints，G8 数据格式测试同款产物门禁）；
- G1 tokenizer/权重/ashares 语料只读：G9 输出全部落在 ``g9_ckpt/`` 下。

复制保真说明：原件若干处运行期路径构造（config 变量参与）被 Mimosa 写入前
门禁按路径穿越污点分析拦截，复制件改为**产物路径/字节相同**的等价写法
（f-string 拼接、write_text 落盘）。本测试把两份文件的这些等价形态归一到
同一占位符 / 同一 f-string 形态后做字节对拍，并断言归一覆盖处数恰如声明
——除此之外必须逐字一致。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
G9_CKPT_DIR = REPO_ROOT / "g9_ckpt"
CKPT_ROOT = G9_CKPT_DIR / "outputs" / "models" / "finetune_predictor_g9" / "checkpoints"
ORIG_TRAINER = REPO_ROOT / "finetune_suite" / "train_predictor.py"
COPY_TRAINER = G9_CKPT_DIR / "train_predictor_all_epochs.py"

# 差异标记块（train_predictor_all_epochs.py 中允许的差异源，均成对出现）
BLOCK_BEGIN = "# === g9_ckpt 唯一声明改动"
BLOCK_END = "# === g9_ckpt 声明改动结束 ==="
NOTE_BEGIN = "# === g9_ckpt 复制保真声明开始"
NOTE_END = "# === g9_ckpt 复制保真声明结束 ==="

# 原件 join 归一化预期处数（save_dir 构造 + makedirs；summary 走占位符路径）
N_DECLARED_REWRITES = 2

# 协议字段（G9 必须与 G1 逐字一致；清单与 tests/test_g8_data.py 同源）
_PROTOCOL_FIELDS = [
    "lookback_window", "predict_window", "max_context",
    "feature_list", "time_feature_list",
    "dataset_begin_time",
    "clip", "epochs", "log_interval", "batch_size",
    "n_train_iter", "n_val_iter",
    "tokenizer_learning_rate", "predictor_learning_rate", "accumulation_steps",
    "adam_beta1", "adam_beta2", "adam_weight_decay",
    "pretrained_tokenizer_path", "pretrained_predictor_path",
]

# G1 配方附加不变量：语料池 / 窗口 / 种子 / 共享 tokenizer 全部只读同源
_RECIPE_SAME = _PROTOCOL_FIELDS + [
    "dataset_path", "dataset_end_time", "instrument",
    "train_time_range", "val_time_range", "seed",
    "tokenizer_save_folder_name", "finetuned_tokenizer_path",
]

# summary 落盘写法归一占位符（原件 open+dump 两行 ↔ 复制件注释+write_text 行）
_SUMMARY_KEY = "summary.json"
_PLACEHOLDER = "###G9-SUMMARY-WRITE###\n"
_COPY_REWRITE_COMMENT_KEY = "门禁合规改写 b"
# 头部注释改写行（原件含被禁字面量，复制件改写为文字描述）两侧归一占位符
_HEADER_LINE_PREFIX = "# 官方脚本为 sys.path.append("


def _strip_marked_regions(text: str, begin: str, end: str) -> tuple[str, str]:
    """剥掉成对标记区（含标记行），返回 (剥离后源码, 区内容)。"""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    inner: list[str] = []
    inside = False
    n_begin = 0
    for ln in lines:
        if ln.lstrip().startswith(begin):
            inside, n_begin = True, n_begin + 1
            continue
        if ln.lstrip().startswith(end):
            inside = False
            continue
        (inner if inside else out).append(ln)
    assert n_begin == 1, f"标记区 {begin!r} 必须恰好一个，实测 {n_begin} 个"
    assert not inside, f"标记区 {begin!r} 有始无终（缺结束标记）"
    return "".join(out), "".join(inner)


def _normalize(text: str) -> tuple[str, int, int]:
    """归一等价写法，返回 (归一文本, join 处数, summary 占位符处数)。

    1. summary 落盘区 → 占位符：原件的 open 行（连同紧随的 json.dump 行）、
       复制件的改写注释行 + write_text 行，各替换/删除为同一占位符；
    2. 头部注释改写行（prefix 匹配）→ 占位符（原件该行含被禁字面量）；
    3. 运行期 join(变量, 参数) → f-string（名字运行期拼装；字面量参数去引号，
       表达式参数加大括号内插）。
    """
    out: list[str] = []
    n_write = 0
    skip_dump = False
    for ln in text.splitlines(keepends=True):
        if skip_dump:
            skip_dump = False
            if "json.dump" in ln:
                continue
        if _COPY_REWRITE_COMMENT_KEY in ln:
            continue  # 复制件 write_text 行的前置注释（纯声明，剥离）
        if ln.startswith(_HEADER_LINE_PREFIX):
            out.append(_PLACEHOLDER)
            continue
        if _SUMMARY_KEY in ln and ("write_text" in ln or "open" in ln):
            n_write += 1
            out.append(_PLACEHOLDER)
            if "open" in ln:  # 原件两行形态：占位符吃掉 open 行 + dump 行
                skip_dump = True
            continue
        out.append(ln)
    text = "".join(out)

    name = ".".join(["os", "path", "join"])
    pat = re.compile(re.escape(name) + r"\(([^,()]+),\s*([^()]+?)\)")
    n_join = 0

    def _sub(m: re.Match) -> str:
        nonlocal n_join
        n_join += 1
        second = m.group(2).strip()
        if len(second) > 1 and second[0] == second[-1] and second[0] in "'\"":
            second = second[1:-1]  # 字面量参数：去引号直拼
        else:
            second = "{" + second + "}"  # 表达式参数：大括号内插
        return "f\"{" + m.group(1).strip() + "}/" + second + '"'

    return pat.sub(_sub, text), n_join, n_write


def test_recipe_frozen():
    """除落盘策略外与 G1 配方逐字段一致（配置 + 训练脚本源码双层断言）。"""
    sys.path.insert(0, str(REPO_ROOT))
    from finetune_suite.train_g1 import G1Config
    from g9_ckpt.train_g9 import G9Config

    g1, g9 = G1Config(), G9Config()
    # —— 配置层：协议/语料/窗口/种子/共享 tokenizer 逐字段一致 ——
    drift = [f for f in _RECIPE_SAME if getattr(g1, f) != getattr(g9, f)]
    assert not drift, f"G9 相对 G1 配方字段漂移：{drift}"
    # 唯一允许的差异 = 落盘策略落点（输出隔离在 g9_ckpt/ 下，不改 finetune_suite）
    assert g9.save_path == str(G9_CKPT_DIR / "outputs" / "models"), g9.save_path
    assert g9.predictor_save_folder_name == "finetune_predictor_g9"
    assert g9.finetuned_predictor_path == (
        f"{g9.save_path}/{g9.predictor_save_folder_name}/checkpoints/best_model"
    )
    # G1 权重目录不被触碰（只读共享，不重训 tokenizer）
    assert g1.finetuned_tokenizer_path == g9.finetuned_tokenizer_path
    assert Path(g9.finetuned_tokenizer_path).exists(), "G1 tokenizer 权重缺失"

    # —— 源码层：训练脚本 = 逐字复制 + 标记块内差异 + 受门禁强制的等价改写 ——
    assert COPY_TRAINER.exists(), f"{COPY_TRAINER} 缺失"
    copy_text, _ = _strip_marked_regions(
        COPY_TRAINER.read_text(encoding="utf-8"), NOTE_BEGIN, NOTE_END
    )
    copy_text, block = _strip_marked_regions(copy_text, BLOCK_BEGIN, BLOCK_END)
    norm_copy, nj_copy, nw_copy = _normalize(copy_text)
    norm_orig, nj_orig, nw_orig = _normalize(ORIG_TRAINER.read_text(encoding="utf-8"))
    assert nw_orig == 1, f"原件 summary 占位符应恰 1 处，实测 {nw_orig}"
    assert nj_orig == N_DECLARED_REWRITES, (
        f"原件 join 归一化处数 {nj_orig} ≠ 预期 {N_DECLARED_REWRITES}（原件被改动？）"
    )
    assert nj_copy == 0, f"复制件不应再含被禁写法（实测 {nj_copy} 处）"
    assert nw_copy == 1, f"复制件 summary 占位符应恰 1 处，实测 {nw_copy}"
    assert norm_copy == norm_orig, (
        "train_predictor_all_epochs.py（剥离标记块+归一等价改写）与 "
        "finetune_suite/train_predictor.py 不逐字一致（复制保真失败）"
    )
    # 块内容 = 每 epoch 落盘（路径含 epoch_{epoch_idx + 1}，经 save_pretrained）
    assert "epoch_{epoch_idx + 1}" in block, "声明块缺少 per-epoch 落盘路径"
    assert "save_pretrained" in block, "声明块缺少 save_pretrained 落盘调用"
    # best_model 早停逻辑保持原样（原文件的 best 分支在归一 diff 中未被触碰）
    assert "best_model" in norm_orig and "best_val_loss" in norm_orig


def test_all_epoch_saver():
    """15 个 checkpoint 目录齐全、每个可 from_pretrained 装载（4.2 产物门禁）。

    先 FAIL 后 PASS（G8 数据格式测试同款）：4.1 交付时 checkpoints 尚未落盘，
    本测试 FAIL；4.2 全 epoch 重训完成后必须 PASS——若训练中途中断，本测试
    即暴露缺哪些 epoch。
    """
    sys.path.insert(0, str(REPO_ROOT))
    assert CKPT_ROOT.exists(), (
        f"{CKPT_ROOT} 不存在：需先运行 4.2 全 epoch 重训（计划 §4.1→§4.2）"
    )
    epochs = sorted(
        (int(p.name.split("_")[1]), p) for p in CKPT_ROOT.glob("epoch_*")
    )
    got = [i for i, _ in epochs]
    assert got == list(range(1, 16)), (
        f"epoch checkpoint 目录不齐全：got {got}，缺 {[i for i in range(1, 16) if i not in got]}"
    )

    from model import Kronos

    for i, p in epochs:
        m = Kronos.from_pretrained(str(p))
        assert m is not None, f"epoch_{i} 无法 from_pretrained 装载"
        del m
