"""
主运行脚本
整合所有模块，提供完整的训练、评估和可视化流程
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# 导入项目模块
from models.baseline import BaselineFraudModel
from models.cdat_fd import DANN
from models.finetune import FineTuneModel

from trainers.trainer_baseline import BaselineTrainer
from trainers.trainer_cdat_fd import CDATFDTrainer
from trainers.trainer_finetune import FineTuneTrainer

from evaluation.evaluation import evaluate_fraud_model, print_metrics
from visualization.tsne_plot import extract_features, plot_tsne

from data import (
    load_source_data,
    load_target_train_data,
    create_dataloader
)


def preprocess_features(features, scaler=None, fit=True):
    """
    特征预处理：标准化
    
    Args:
        features: 特征数组（numpy array）
        scaler: 已训练的标准化器（可选）
        fit: 是否拟合新的标准化器
    
    Returns:
        处理后的特征和标准化器
    """
    # 确保是numpy数组且为数值类型
    if isinstance(features, pd.DataFrame):
        features = features.values
    
    # 转换为float类型，处理非数值
    features = pd.DataFrame(features).apply(pd.to_numeric, errors='coerce').values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
    
    # 再次处理NaN和Inf（标准化后可能产生）
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features, scaler


def align_features(source_features, target_features, method='truncate'):
    """
    对齐源域和目标域的特征维度
    
    Args:
        source_features: 源域特征
        target_features: 目标域特征
        method: 对齐方法 ('truncate' 或 'pca')
    
    Returns:
        对齐后的特征和特征维度
    """
    src_dim = source_features.shape[1]
    tgt_dim = target_features.shape[1]
    
    if src_dim == tgt_dim:
        return source_features, target_features, src_dim
    
    # 选择较小的维度作为目标维度
    target_dim = min(src_dim, tgt_dim)
    
    if method == 'pca' and src_dim != tgt_dim:
        # 使用PCA降维（保留更多信息）
        from sklearn.decomposition import PCA
        
        print(f"[INFO] 使用PCA对齐特征维度: 源域 {src_dim} -> {target_dim}, 目标域 {tgt_dim} -> {target_dim}")
        
        # 合并数据找到共同的主成分
        combined = np.vstack([source_features, target_features])
        pca = PCA(n_components=target_dim, random_state=42)
        pca.fit(combined)
        
        source_aligned = pca.transform(source_features)
        target_aligned = pca.transform(target_features)
        
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"[INFO] PCA解释方差比例: {explained_var:.4f}")
        
        return source_aligned, target_aligned, target_dim
    else:
        # 简单截断（默认方法）
        if src_dim > target_dim:
            source_features = source_features[:, :target_dim]
        if tgt_dim > target_dim:
            target_features = target_features[:, :target_dim]
        
        print(f"[INFO] 特征维度对齐（截断）: 源域 {src_dim} -> {target_dim}, 目标域 {tgt_dim} -> {target_dim}")
        return source_features, target_features, target_dim


def train_baseline(source_features, source_labels, device, config):
    """训练Baseline模型"""
    print("\n" + "="*60)
    print("训练 Baseline 模型")
    print("="*60)
    
    # 数据预处理
    source_features, scaler = preprocess_features(source_features, fit=True)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        source_features, source_labels,
        test_size=0.2,
        random_state=42,
        stratify=source_labels
    )
    
    # 创建DataLoader
    train_loader = create_dataloader(
        X_train, y_train,
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_loader = create_dataloader(
        X_val, y_val,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # 创建模型
    input_dim = source_features.shape[1]
    model = BaselineFraudModel(input_dim=input_dim)
    
    # 训练
    trainer = BaselineTrainer(
        model=model,
        train_loader=train_loader,
        device=device,
        lr=config['lr']
    )
    trainer.train(epochs=config['epochs'])
    
    # 评估
    print("\n[评估] Baseline模型在验证集上的表现:")
    metrics = evaluate_fraud_model(model, val_loader, device=device, find_optimal_threshold=True)
    print_metrics(metrics, prefix="  ", verbose=True)
    
    return model, scaler


def train_cdat_fd(source_features, source_labels, target_features, device, config):
    """训练CDAT-FD（域适应）模型"""
    print("\n" + "="*60)
    print("训练 CDAT-FD (域适应) 模型")
    print("="*60)
    
    # 特征对齐（使用PCA保留更多信息）
    source_features, target_features, input_dim = align_features(
        source_features, target_features, method='pca'
    )
    
    # 数据预处理
    source_features, scaler = preprocess_features(source_features, fit=True)
    target_features, _ = preprocess_features(target_features, scaler=scaler, fit=False)
    
    # 划分目标域数据（用于训练）
    # 注意：目标域通常没有标签，这里假设有部分标签用于训练
    # 如果没有标签，可以只用源域标签训练
    if len(target_features) > 1000:
        # 如果目标域数据量大，取一部分用于训练
        target_train_features = target_features[:len(target_features)//2]
    else:
        target_train_features = target_features
    
    # 创建DataLoader
    source_loader = create_dataloader(
        source_features, source_labels,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    # 目标域数据（无标签，只用于域适应）
    target_loader = create_dataloader(
        target_train_features, None,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    # 创建模型
    model = DANN(src_input_dim=input_dim, tgt_input_dim=input_dim)
    
    # 训练
    trainer = CDATFDTrainer(
        model=model,
        src_loader=source_loader,
        tgt_loader=target_loader,
        device=device,
        lr=config['lr']
    )
    trainer.train(num_epochs=config['epochs'])
    
    # 在源域验证集上评估
    X_train, X_val, y_train, y_val = train_test_split(
        source_features, source_labels,
        test_size=0.2,
        random_state=42,
        stratify=source_labels
    )
    val_loader = create_dataloader(X_val, y_val, batch_size=config['batch_size'], shuffle=False)
    
    print("\n[评估] CDAT-FD模型在源域验证集上的表现:")
    metrics = evaluate_fraud_model(model, val_loader, device=device, find_optimal_threshold=True)
    print_metrics(metrics, prefix="  ", verbose=True)
    
    return model, scaler


def train_finetune(source_model, target_features, target_labels, device, config):
    """在目标域上微调模型"""
    print("\n" + "="*60)
    print("Fine-tune 模型（在目标域上微调）")
    print("="*60)
    
    # 创建FineTune模型（使用源域模型的encoder）
    input_dim = target_features.shape[1]
    finetune_model = FineTuneModel(input_dim=input_dim)
    
    # 如果源域模型有encoder，可以加载权重
    if hasattr(source_model, 'encoder'):
        try:
            finetune_model.encoder.load_state_dict(source_model.encoder.state_dict())
            print("[INFO] 已加载源域模型的encoder权重")
        except:
            print("[WARN] 无法加载源域encoder权重，使用随机初始化")
    
    # 数据预处理
    target_features, scaler = preprocess_features(target_features, fit=True)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        target_features, target_labels,
        test_size=0.2,
        random_state=42,
        stratify=target_labels if len(np.unique(target_labels)) > 1 else None
    )
    
    # 创建DataLoader
    train_loader = create_dataloader(
        X_train, y_train,
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_loader = create_dataloader(
        X_val, y_val,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # 训练
    trainer = FineTuneTrainer(
        model=finetune_model,
        train_loader=train_loader,
        device=device,
        lr=config['lr'] * 0.1,  # 微调使用更小的学习率
        freeze_encoder=False
    )
    trainer.train(epochs=config['finetune_epochs'])
    
    # 评估
    print("\n[评估] Fine-tune模型在目标域验证集上的表现:")
    metrics = evaluate_fraud_model(finetune_model, val_loader, device=device, find_optimal_threshold=True)
    print_metrics(metrics, prefix="  ", verbose=True)
    
    return finetune_model, scaler


def visualize_features(baseline_model, dann_model, source_loader, target_loader, device, save_dir):
    """可视化特征分布（t-SNE）"""
    print("\n" + "="*60)
    print("生成 t-SNE 可视化")
    print("="*60)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Baseline模型的特征
    print("[1] 提取Baseline模型特征...")
    src_features_baseline, _ = extract_features(baseline_model, source_loader, device, max_samples=2000)
    tgt_features_baseline, _ = extract_features(baseline_model, target_loader, device, max_samples=2000)
    
    features_baseline = np.vstack([src_features_baseline, tgt_features_baseline])
    domain_labels_baseline = np.array([0] * len(src_features_baseline) + [1] * len(tgt_features_baseline))
    
    plot_tsne(
        features_baseline,
        domain_labels_baseline,
        title="t-SNE Visualization (Baseline - Before Alignment)",
        save_path=str(save_dir / "tsne_baseline.png")
    )
    
    # DANN模型的特征
    print("[2] 提取DANN模型特征...")
    src_features_dann, _ = extract_features(dann_model, source_loader, device, max_samples=2000)
    tgt_features_dann, _ = extract_features(dann_model, target_loader, device, max_samples=2000)
    
    features_dann = np.vstack([src_features_dann, tgt_features_dann])
    domain_labels_dann = np.array([0] * len(src_features_dann) + [1] * len(tgt_features_dann))
    
    plot_tsne(
        features_dann,
        domain_labels_dann,
        title="t-SNE Visualization (DANN - After Alignment)",
        save_path=str(save_dir / "tsne_dann.png")
    )
    
    print(f"[OK] 可视化结果已保存到: {save_dir}")


def save_models(baseline_model, dann_model, finetune_model, save_dir):
    """保存训练好的模型"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("保存模型")
    print("="*60)
    
    saved_count = 0
    
    # 保存Baseline模型
    if baseline_model is not None:
        baseline_path = save_dir / "baseline_model.pth"
        torch.save(baseline_model.state_dict(), baseline_path)
        print(f"[OK] Baseline模型已保存: {baseline_path}")
        saved_count += 1
    
    # 保存DANN模型
    if dann_model is not None:
        dann_path = save_dir / "dann_model.pth"
        torch.save(dann_model.state_dict(), dann_path)
        print(f"[OK] DANN模型已保存: {dann_path}")
        saved_count += 1
    
    # 保存FineTune模型
    if finetune_model is not None:
        finetune_path = save_dir / "finetune_model.pth"
        torch.save(finetune_model.state_dict(), finetune_path)
        print(f"[OK] FineTune模型已保存: {finetune_path}")
        saved_count += 1
    
    if saved_count == 0:
        print("[WARN] 没有模型需要保存")
    else:
        print(f"[OK] 共保存了 {saved_count} 个模型")


def main():
    parser = argparse.ArgumentParser(description='欺诈检测域适应项目主脚本')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['baseline', 'cdat_fd', 'finetune', 'all'],
                       help='训练模式: baseline, cdat_fd, finetune, 或 all')
    parser.add_argument('--source_data', type=str,
                       default='data/creditcard/creditcard.csv',
                       help='源域数据路径')
    parser.add_argument('--target_train_trans', type=str,
                       default='data/ieee_fraud/train_transaction.csv',
                       help='目标域训练交易数据路径')
    parser.add_argument('--target_train_id', type=str,
                       default='data/ieee_fraud/train_identity.csv',
                       help='目标域训练身份数据路径（可选）')
    parser.add_argument('--epochs', type=int, default=20,
                       help='训练轮数')
    parser.add_argument('--finetune_epochs', type=int, default=10,
                       help='微调轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='计算设备')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='模型保存目录')
    parser.add_argument('--visualize', action='store_true',
                       help='是否生成t-SNE可视化')
    parser.add_argument('--vis_dir', type=str, default='visualizations',
                       help='可视化结果保存目录')
    
    args = parser.parse_args()
    
    # 设备配置
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("="*60)
    print("欺诈检测域适应项目")
    print("="*60)
    print(f"设备: {device}")
    print(f"训练模式: {args.mode}")
    print("="*60)
    
    # 配置
    config = {
        'epochs': args.epochs,
        'finetune_epochs': args.finetune_epochs,
        'batch_size': args.batch_size,
        'lr': args.lr
    }
    
    # 加载数据
    print("\n[1] 加载数据...")
    try:
        source_features, source_labels = load_source_data(args.source_data)
        print(f"[OK] 源域数据加载成功: {source_features.shape[0]} 样本, {source_features.shape[1]} 特征")
    except Exception as e:
        print(f"[ERROR] 源域数据加载失败: {e}")
        return
    
    try:
        if Path(args.target_train_trans).exists():
            try:
                target_features, target_labels = load_target_train_data(
                    args.target_train_trans,
                    args.target_train_id if Path(args.target_train_id).exists() else None
                )
                print(f"[OK] 目标域数据加载成功: {target_features.shape[0]} 样本, {target_features.shape[1]} 特征")
            except Exception as e:
                print(f"[WARN] 目标域数据加载失败（可能没有标签）: {e}")
                # 尝试加载测试数据作为无标签数据
                from data import load_target_test_data
                target_features, _ = load_target_test_data(
                    args.target_train_trans,
                    args.target_train_id if Path(args.target_train_id).exists() else None
                )
                target_labels = np.zeros(len(target_features))  # 伪标签
                print(f"[INFO] 使用测试数据作为目标域（无标签）: {target_features.shape[0]} 样本")
        else:
            raise FileNotFoundError(f"目标域数据文件不存在: {args.target_train_trans}")
    except Exception as e:
        print(f"[ERROR] 目标域数据加载失败: {e}")
        print("[INFO] 将只训练Baseline模型（不需要目标域数据）")
        target_features = None
        target_labels = None
    
    # 训练模型
    baseline_model = None
    dann_model = None
    finetune_model = None
    
    if args.mode in ['baseline', 'all']:
        baseline_model, _ = train_baseline(source_features, source_labels, device, config)
    
    if args.mode in ['cdat_fd', 'all']:
        if target_features is not None:
            dann_model, _ = train_cdat_fd(source_features, source_labels, target_features, device, config)
        else:
            print("[WARN] 目标域数据不可用，跳过CDAT-FD训练")
            dann_model = None
    
    if args.mode in ['finetune', 'all']:
        if target_features is None or target_labels is None:
            print("[WARN] 目标域数据不可用，跳过FineTune训练")
            finetune_model = None
        elif baseline_model is None and dann_model is None:
            print("[WARN] 需要先训练baseline或cdat_fd模型才能进行微调")
            finetune_model = None
        else:
            source_model = dann_model if dann_model is not None else baseline_model
            finetune_model, _ = train_finetune(source_model, target_features, target_labels, device, config)
    
    # 可视化
    if args.visualize and baseline_model is not None and dann_model is not None:
        # 创建用于可视化的DataLoader
        source_vis_loader = create_dataloader(
            source_features[:2000], source_labels[:2000],
            batch_size=args.batch_size, shuffle=False
        )
        target_vis_loader = create_dataloader(
            target_features[:2000], None,
            batch_size=args.batch_size, shuffle=False
        )
        visualize_features(
            baseline_model, dann_model,
            source_vis_loader, target_vis_loader,
            device, args.vis_dir
        )
    
    # 保存模型
    if baseline_model is not None or dann_model is not None or finetune_model is not None:
        save_models(baseline_model, dann_model, finetune_model, args.save_dir)
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)


if __name__ == "__main__":
    main()

