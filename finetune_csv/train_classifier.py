"""方向分类训练脚本（单脚本支持方案 A / 方案 B 切换）。

按 config 的 ``classifier.backbone`` 字段切换：

- ``from_scratch``：训练 :class:`KronosClassifier`（从头训练判别式主干）；
- ``pretrained_probe``：冻结预训练 Kronos 主干，仅训练 linear probe（对照实验）。

两类模型共享统一的 ``forward(features, stamp) -> logits`` 接口，故数据集、训练循环、
评估与基线对比代码完全复用，保证 A/B 对照可比性。

训练目标：按 val macro-F1 早停选模型；测试阶段输出 accuracy/macro-F1/混淆矩阵，
并与「多数类基线」「动量基线」对比，金融口径给出多空累计收益与胜率。

设备：cuda → mps → cpu 三级回退（参考方案文档 4.1 节 Mac MPS 实测结论）。
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config_loader import ClassifierConfig
from dataset_classification import build_classification_datasets
from model import Kronos, KronosClassifier, KronosProbeClassifier, KronosTokenizer


# ---------------------------------------------------------------------------
# 设备与随机性
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """三级设备回退：cuda → mps → cpu。

    :return: 可用 torch 设备。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """固定各随机源。

    :param seed: 随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 模型构建
# ---------------------------------------------------------------------------
def build_model(config: ClassifierConfig, device: torch.device) -> nn.Module:
    """按 backbone 字段构建模型。

    :param config: 分类配置。
    :param device: 目标设备。
    :return: 已迁移到设备的模型。
    """
    if config.backbone == "from_scratch":
        model = KronosClassifier(
            d_in=config.d_in,
            d_model=config.d_model,
            n_heads=config.n_heads,
            ff_dim=config.ff_dim,
            n_layers=config.n_layers,
            n_classes=config.n_classes,
            learn_te=config.learn_te,
            dropout=config.dropout,
        )
        return model.to(device)

    # 方案 B：pretrained_probe
    tokenizer, kronos = _load_probe_backbone(config)
    tokenizer = tokenizer.to(device)
    kronos = kronos.to(device)
    model = KronosProbeClassifier(tokenizer, kronos, n_classes=config.n_classes).to(device)
    return model


def _load_probe_backbone(config: ClassifierConfig) -> tuple[nn.Module, nn.Module]:
    """加载方案 B 的冻结主干。

    从预训练路径加载 Tokenizer 与 Kronos 主干。**路径为空时直接报错**——方案 B 的意义
    在于继承预训练红利，无预训练主干的随机初始化对照无价值，不应静默兜底。

    :param config: 分类配置。
    :return: (tokenizer, kronos)。
    :raises ValueError: 预训练 tokenizer 或 predictor 路径未配置。
    """
    tok_path = config.pretrained_tokenizer
    pred_path = config.pretrained_predictor

    if not tok_path or not pred_path:
        raise ValueError(
            "pretrained_probe 模式必须提供预训练权重路径："
            f"pretrained_tokenizer='{tok_path}', pretrained_predictor='{pred_path}'。"
            "请在 config 中填入（如 NeoQuasar/Kronos-Tokenizer-base 与 NeoQuasar/Kronos-base）。"
        )

    print(f"加载预训练 Tokenizer: {tok_path}")
    print(f"加载预训练 Predictor: {pred_path}")
    tokenizer = KronosTokenizer.from_pretrained(tok_path)
    kronos = Kronos.from_pretrained(pred_path)
    return tokenizer, kronos


# ---------------------------------------------------------------------------
# 评估指标（纯 numpy 手算，不引入 sklearn）
# ---------------------------------------------------------------------------
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    """计算 macro-F1（各类 F1 的算术平均）。

    :param y_true: 真实标签。
    :param y_pred: 预测标签。
    :param n_classes: 类别数。
    :return: macro-F1。
    """
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """计算混淆矩阵，行=真实，列=预测。

    :param y_true: 真实标签。
    :param y_pred: 预测标签。
    :param n_classes: 类别数。
    :return: ``[n_classes, n_classes]`` 整数矩阵。
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def momentum_baseline(
    close: np.ndarray, t_idx: np.ndarray, dataset, h: int
) -> np.ndarray:
    """动量基线：``close[t]/close[t-H] - 1`` 用**与标签同源的冻结阈值**分箱。

    不再 hardcode ``>0/<0``，而是复用 :meth:`KlineClassificationDataset._rate_to_label`
    的 frozen low/high 阈值，保证动量基线与真实标签口径一致、可比。

    :param close: test 段 close 序列。
    :param t_idx: 每个样本对应的 t 下标（在 close 中的位置）。
    :param dataset: 数据集对象（取冻结阈值与分箱方法）。
    :param h: 回看步数（用 predict_window）。
    :return: 动量基线预测标签。
    """
    past = close[t_idx - h]
    now = close[t_idx]
    ret = now / past - 1.0
    return dataset._rate_to_label(ret)


# ---------------------------------------------------------------------------
# 训练 / 评估循环
# ---------------------------------------------------------------------------
def evaluate(model, loader, device, class_weights=None) -> tuple[float, float, np.ndarray, np.ndarray]:
    """在给定 loader 上评估。

    :param model: 模型。
    :param loader: 数据加载器。
    :param device: 设备。
    :param class_weights: 可选的交叉熵权重（仅用于计算加权 loss）。
    :return: (avg_loss, macro_f1, y_true, y_pred)。
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    total_loss = 0.0
    n = 0
    ys_true, ys_pred = [], []
    with torch.no_grad():
        for features, stamp, labels in loader:
            features = features.to(device)
            stamp = stamp.to(device)
            labels = labels.to(device)
            logits = model(features, stamp)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            n += labels.size(0)
            ys_pred.append(logits.argmax(dim=-1).cpu().numpy())
            ys_true.append(labels.cpu().numpy())
    y_true = np.concatenate(ys_true) if ys_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(ys_pred) if ys_pred else np.array([], dtype=np.int64)
    avg_loss = total_loss / max(n, 1)
    f1 = macro_f1(y_true, y_pred)
    return avg_loss, f1, y_true, y_pred


def train(config: ClassifierConfig) -> dict:
    """完整训练 + 评估流程。

    :param config: 分类配置。
    :return: 结果字典（含 test 指标与基线对比）。
    """
    set_seed(config.seed)
    device = get_device()
    print(f"使用设备：{device}")
    config.print_config_summary()

    # 数据
    train_ds, val_ds, test_ds = build_classification_datasets(config)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    # 模型
    model = build_model(config, device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数：{n_params:,}（可训练 {n_trainable:,}）")

    # 损失：按训练集类别频率倒数加权
    class_weights = None
    if config.class_weighted:
        w = train_ds.compute_class_weights()
        class_weights = torch.from_numpy(w).float().to(device)
        print(f"class_weights: {w}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # 只优化可训练参数
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.lr,
        steps_per_epoch=max(len(train_loader), 1),
        epochs=config.epochs,
        pct_start=0.03,
        div_factor=10,
    )

    # 训练循环 + 按 val macro-F1 早停
    best_f1 = -1.0
    best_state = None
    patience = 0
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / f"best_{config.backbone}.pt"

    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        nb = 0
        for i, (features, stamp, labels) in enumerate(train_loader):
            features = features.to(device)
            stamp = stamp.to(device)
            labels = labels.to(device)
            logits = model(features, stamp)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=config.grad_clip
            )
            optimizer.step()
            scheduler.step()
            running += loss.item()
            nb += 1
            if (i + 1) % config.log_interval == 0:
                print(f"[Epoch {epoch+1}/{config.epochs} Step {i+1}/{len(train_loader)}] loss={loss.item():.4f}")

        val_loss, val_f1, _, _ = evaluate(model, val_loader, device, class_weights)
        print(f"--- Epoch {epoch+1} train_loss={running/max(nb,1):.4f} val_loss={val_loss:.4f} val_macroF1={val_f1:.4f} ---")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, best_path)
            print(f"  → 新最优 val macro-F1={best_f1:.4f}，保存到 {best_path}")
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stop_patience:
                print(f"早停：val macro-F1 连续 {patience} 轮无提升")
                break

    # 恢复最优权重做 test
    if best_state is not None:
        model.load_state_dict(best_state)

    results = _test_and_compare(model, test_loader, test_ds, train_ds, device, config)
    return results


def _test_and_compare(model, test_loader, test_ds, train_ds, device, config: ClassifierConfig) -> dict:
    """测试阶段：分类口径 + 双基线对比 + 金融口径。

    :param model: 已加载最优权重的模型。
    :param test_loader: 测试集加载器。
    :param test_ds: 测试集对象（用于 close 序列与下标）。
    :param train_ds: 训练集对象（用于估计多数类基线标签）。
    :param device: 设备。
    :param config: 分类配置。
    :return: 结果字典。
    """
    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)

    test_loss, test_f1, y_true, y_pred = evaluate(model, test_loader, device)
    acc = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    cm = confusion_matrix(y_true, y_pred)
    print(f"accuracy={acc:.4f} macro-F1={test_f1:.4f}")
    print("混淆矩阵（行=真实 列=预测，0=跌 1=平 2=涨）：")
    print(cm)

    # 基线 1：多数类——用训练集多数类标签填充（而非 test 集自身多数类）
    train_labels = train_ds._collect_labels()
    maj = int(np.argmax(np.bincount(train_labels, minlength=3))) if len(train_labels) else 1
    maj_acc = float(np.mean(y_true == maj)) if len(y_true) else 0.0
    maj_f1 = macro_f1(y_true, np.full_like(y_true, maj))

    # 基线 2：动量 close[t]/close[t-H]-1，用与标签同源的冻结阈值分箱
    close = test_ds.close_series
    l, h = config.lookback_window, config.predict_window
    # 每个样本 t 在 test_ds 内部的下标
    t_idx = np.arange(test_ds.n_samples) + l - 1
    # 动量需 close[t-h]，t-h >= 0 即 t_idx >= h；test 段 t_idx 从 l-1 起，通常 l > h
    if test_ds.n_samples > 0 and (l - 1) >= h:
        mom_pred = momentum_baseline(close, t_idx, test_ds, h)
        mom_acc = float(np.mean(y_true == mom_pred))
        mom_f1 = macro_f1(y_true, mom_pred)
    else:
        mom_pred = None
        mom_acc = float("nan")
        mom_f1 = float("nan")

    print("\n--- 基线对比 ---")
    print(f"{'方法':<20}{'accuracy':<12}{'macro-F1':<12}")
    print(f"{'多数类基线':<18}{maj_acc:<12.4f}{maj_f1:<12.4f}")
    print(f"{'动量基线':<18}{mom_acc:<12.4f}{mom_f1:<12.4f}")
    print(f"{'本模型':<18}{acc:<12.4f}{test_f1:<12.4f}")

    # 警示：若未战胜基线
    if test_f1 < maj_f1:
        print("⚠ 模型 macro-F1 未超过多数类基线，说明未学到有效信号，勿据 accuracy 下结论")
    if not np.isnan(mom_f1) and test_f1 < mom_f1:
        print("⚠ 模型 macro-F1 未超过动量基线")

    # 金融口径：按预测方向多空，不重叠采样后的平均区间收益与胜率
    fin = _financial_metric(y_pred, test_ds, config)

    return {
        "accuracy": acc,
        "macro_f1": test_f1,
        "confusion_matrix": cm.tolist(),
        "majority": {"label": maj, "accuracy": maj_acc, "macro_f1": maj_f1},
        "momentum": {"accuracy": mom_acc, "macro_f1": mom_f1},
        "financial": fin,
    }


def _financial_metric(y_pred, test_ds, config: ClassifierConfig) -> dict:
    """金融口径：按预测方向多空，不重叠区间的平均每步收益与胜率。

    原始样本逐点滚动，相邻样本的未来 H 步窗口重叠 H-1 步，累计收益会重复计数。
    改为步长 H 不重叠采样：取 i = 0, H, 2H, ... 的样本，每个区间长度恰为 H，
    区间收益除以 H 得「平均每步收益」，再按预测方向带符号汇总。

    :param y_pred: 模型预测标签（覆盖全部 test 样本）。
    :param test_ds: 测试集对象。
    :param config: 配置。
    :return: 金融指标字典。
    """
    close = test_ds.close_series
    l, h = config.lookback_window, config.predict_window
    n = test_ds.n_samples
    if n == 0:
        return {"mean_period_return": 0.0, "win_rate": 0.0, "n_periods": 0}

    # 不重叠采样：样本下标 0, H, 2H, ...；对应 t = i+L-1，区间 [t, t+H]
    sample_idx = np.arange(0, n, h)
    t_idx = sample_idx + l - 1
    rets = close[t_idx + h] / close[t_idx] - 1.0
    preds = y_pred[sample_idx]
    # 方向仓位：涨→+1，跌→-1，平→0
    pos = np.where(preds == 2, 1.0, np.where(preds == 0, -1.0, 0.0))
    pnl = pos * rets
    # 区间收益除以 H 折算成平均每步收益
    mean_ret = float(pnl.sum() / (len(sample_idx) * h))
    traded = np.abs(pos) > 0
    win = float(np.mean(pnl[traded] > 0)) if traded.any() else 0.0
    print("\n--- 金融口径（test 段不重叠多空）---")
    print(f"平均每步收益={mean_ret:.6f} 胜率={win:.4f} 交易区间数={int(traded.sum())}/{len(sample_idx)}")
    return {"mean_period_return": mean_ret, "win_rate": win, "n_periods": int(len(sample_idx))}


def main():
    parser = argparse.ArgumentParser(description="Kronos 方向分类训练（方案 A/B 切换）")
    parser.add_argument("--config", type=str, required=True, help="yaml 配置文件路径")
    args = parser.parse_args()

    config = ClassifierConfig(args.config)
    train(config)


if __name__ == "__main__":
    main()
