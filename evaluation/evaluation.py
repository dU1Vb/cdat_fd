import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve
)

@torch.no_grad()
def evaluate_fraud_model(
    model,
    dataloader,
    device="cuda",
    threshold=None,
    find_optimal_threshold=True
):
    """
    对欺诈检测模型进行评估
    返回常用金融风控指标
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        threshold: 分类阈值，如果为None则自动寻找最优阈值
        find_optimal_threshold: 是否自动寻找最优阈值（基于F1）
    
    Returns:
        评估指标字典
    """
    model.eval()

    y_true = []
    y_score = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        # 兼容不同模型：DANCE返回(preds, domain_pred, features)，DANN返回(preds, domain_pred)，Baseline只返回preds
        # 检查模型forward方法是否接受lambda_参数
        import inspect
        sig = inspect.signature(model.forward)
        if 'lambda_' in sig.parameters:
            output = model(x, lambda_=0.0)  # DANCE/DANN模型
        else:
            output = model(x)  # Baseline模型
        
        if isinstance(output, tuple):
            if len(output) == 3:  # DANCE模型
                preds, _, _ = output
            else:  # DANN模型
                preds, _ = output
        else:
            preds = output
        preds = preds.view(-1)

        y_true.extend(y.cpu().numpy())
        y_score.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # 诊断信息
    pos_ratio = np.mean(y_true)
    pred_min = np.min(y_score)
    pred_max = np.max(y_score)
    pred_mean = np.mean(y_score)
    pred_median = np.median(y_score)
    
    # 如果阈值未指定，自动寻找最优阈值
    if threshold is None or find_optimal_threshold:
        # 使用F1分数寻找最优阈值
        best_threshold = 0.5
        best_f1 = 0
        
        # 检查预测分数分布
        pred_min = np.min(y_score)
        pred_max = np.max(y_score)
        
        # 如果预测分数普遍偏低（最大值<0.5），使用更合适的阈值范围
        if pred_max < 0.5:
            # 使用更小的阈值范围
            thresholds = np.arange(0.0001, 0.5, 0.001)
        else:
            # 正常阈值范围
            thresholds = np.arange(0.01, 0.99, 0.01)
        
        for t in thresholds:
            y_pred_t = (y_score >= t).astype(int)
            if np.sum(y_pred_t) > 0:  # 至少有一个正预测
                f1_t = f1_score(y_true, y_pred_t, zero_division=0)
                if f1_t > best_f1:
                    best_f1 = f1_t
                    best_threshold = t
        
        threshold = best_threshold if find_optimal_threshold else (threshold or 0.5)
    
    # 二值预测
    y_pred = (y_score >= threshold).astype(int)

    # 指标计算
    auc = roc_auc_score(y_true, y_score)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # KS 值
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_score)
    ks = np.max(tpr - fpr)

    # 统计信息
    n_pos_pred = np.sum(y_pred)
    n_pos_true = np.sum(y_true)

    return {
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "KS": ks,
        "Threshold": threshold,
        "N_Positive_Predicted": int(n_pos_pred),
        "N_Positive_True": int(n_pos_true),
        "Positive_Ratio": pos_ratio,
        "Pred_Score_Range": (float(pred_min), float(pred_max)),
        "Pred_Score_Mean": float(pred_mean),
        "Pred_Score_Median": float(pred_median)
    }

def print_metrics(metrics, prefix="", verbose=False):
    """
    打印评估指标
    
    Args:
        metrics: 指标字典
        prefix: 前缀字符串
        verbose: 是否显示详细信息
    """
    print(
        f"{prefix}"
        f"AUC: {metrics['AUC']:.4f} | "
        f"Recall: {metrics['Recall']:.4f} | "
        f"Precision: {metrics['Precision']:.4f} | "
        f"F1: {metrics['F1']:.4f} | "
        f"KS: {metrics['KS']:.4f}"
    )
    
    if verbose or metrics.get('Threshold') is not None:
        print(
            f"{prefix}"
            f"最优阈值: {metrics.get('Threshold', 0.5):.4f} | "
            f"预测正样本数: {metrics.get('N_Positive_Predicted', 0)} | "
            f"真实正样本数: {metrics.get('N_Positive_True', 0)} | "
            f"正样本比例: {metrics.get('Positive_Ratio', 0):.4f}"
        )
        if 'Pred_Score_Range' in metrics:
            print(
                f"{prefix}"
                f"预测分数范围: [{metrics['Pred_Score_Range'][0]:.4f}, {metrics['Pred_Score_Range'][1]:.4f}] | "
                f"均值: {metrics['Pred_Score_Mean']:.4f} | "
                f"中位数: {metrics['Pred_Score_Median']:.4f}"
            )
