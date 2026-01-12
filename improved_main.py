"""
改进的主训练脚本，支持更全面的评估
"""
import torch
import numpy as np
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.baseline import BaselineFraudModel
from models.cdat_fd import DANCE
from models.finetune import FineTuneModel
from trainers.trainer_baseline import BaselineTrainer
from trainers.trainer_cdat_fd import CDATFDTrainer
from trainers.trainer_finetune import FineTuneTrainer
from data.loaders import load_source_data, load_target_train_data, load_target_test_data, create_dataloader
from data.preprocessing import preprocess_features, align_features
from evaluation.evaluation import evaluate_fraud_model, print_metrics
from visualization.tsne_plot import visualize_features
from utils import save_models


def train_and_evaluate_baseline(source_features, source_labels, target_features, target_labels, device, config, evaluate_on_target=True):
    """训练并评估Baseline模型"""
    print("\n" + "="*60)
    print("训练 Baseline 模型")
    print("="*60)
    
    # 保存原始源域特征，用于后续对齐
    source_features_orig = source_features.copy()
    
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
    
    # 在源域验证集上评估
    print("\n[评估] Baseline模型在源域验证集上的表现:")
    source_metrics = evaluate_fraud_model(model, val_loader, device=device, find_optimal_threshold=True)
    print_metrics(source_metrics, prefix="  ", verbose=True)
    
    # 在目标域上评估（如果需要）
    target_metrics = None
    if evaluate_on_target and target_features is not None:
        # 特征对齐（使用原始源域特征）
        source_features_aligned, target_features_aligned, _ = align_features(
            source_features_orig, target_features, method='pca'
        )
        
        # 预处理对齐后的特征
        source_features_aligned, _ = preprocess_features(source_features_aligned, scaler=scaler, fit=False)
        target_features_aligned, _ = preprocess_features(target_features_aligned, scaler=scaler, fit=False)
        
        # 创建目标域数据加载器
        target_loader = create_dataloader(
            target_features_aligned, target_labels,
            batch_size=config['batch_size'], shuffle=False
        )
        
        # 在目标域上评估
        print("\n[评估] Baseline模型在目标域上的表现:")
        target_metrics = evaluate_fraud_model(model, target_loader, device=device, find_optimal_threshold=True)
        print_metrics(target_metrics, prefix="  ", verbose=True)
    
    return model, scaler, source_metrics, target_metrics


def train_and_evaluate_dann(source_features, source_labels, target_features, target_labels, device, config, evaluate_on_target=True):
    """训练并评估DANN模型"""
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
    if len(target_features) > 1000:
        target_train_features = target_features[:len(target_features)//2]
    else:
        target_train_features = target_features
    
    # 创建DataLoader
    source_loader = create_dataloader(
        source_features, source_labels,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    target_loader = create_dataloader(
        target_train_features, None,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    # 创建模型
    model = DANCE(src_input_dim=input_dim, tgt_input_dim=input_dim)
    
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
    
    print("\n[评估] DANN模型在源域验证集上的表现:")
    source_metrics = evaluate_fraud_model(model, val_loader, device=device, find_optimal_threshold=True)
    print_metrics(source_metrics, prefix="  ", verbose=True)
    
    # 在目标域上评估（如果需要）
    target_metrics = None
    if evaluate_on_target:
        # 创建目标域数据加载器
        target_eval_loader = create_dataloader(
            target_features, target_labels,
            batch_size=config['batch_size'], shuffle=False
        )
        
        # 在目标域上评估
        print("\n[评估] DANN模型在目标域上的表现:")
        target_metrics = evaluate_fraud_model(model, target_eval_loader, device=device, find_optimal_threshold=True)
        print_metrics(target_metrics, prefix="  ", verbose=True)
    
    return model, scaler, source_metrics, target_metrics


def train_and_evaluate_finetune(source_model, source_features, target_features, target_labels, device, config):
    """在目标域上微调并评估模型"""
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
    
    # 在目标域验证集上评估
    print("\n[评估] Fine-tune模型在目标域验证集上的表现:")
    target_metrics = evaluate_fraud_model(finetune_model, val_loader, device=device, find_optimal_threshold=True)
    print_metrics(target_metrics, prefix="  ", verbose=True)
    
    # 注意：Fine-tune模型是在目标域上训练的，不适合在源域上评估
    # 因为模型的输入层已经针对目标域的432个特征进行了设计
    source_metrics = None
    print("\n[INFO] Fine-tune模型是在目标域上训练的，跳过源域评估")
    
    return finetune_model, scaler, source_metrics, target_metrics


def compare_models(baseline_metrics, dann_metrics, finetune_metrics):
    """比较不同模型的性能"""
    print("\n" + "="*60)
    print("模型性能对比")
    print("="*60)
    
    # 目标域性能对比
    print("\n[目标域性能对比]")
    print("-" * 40)
    
    print(f"{'模型':<12} {'AUC':<8} {'Precision':<12} {'Recall':<8} {'F1':<8}")
    print("-" * 50)
    
    if baseline_metrics is not None:
        print(f"{'Baseline':<12} {baseline_metrics['AUC']:<8.4f} {baseline_metrics['Precision']:<12.4f} "
              f"{baseline_metrics['Recall']:<8.4f} {baseline_metrics['F1']:<8.4f}")
    
    if dann_metrics is not None:
        print(f"{'DANN':<12} {dann_metrics['AUC']:<8.4f} {dann_metrics['Precision']:<12.4f} "
              f"{dann_metrics['Recall']:<8.4f} {dann_metrics['F1']:<8.4f}")
    
    if finetune_metrics is not None:
        print(f"{'Fine-tune':<12} {finetune_metrics['AUC']:<8.4f} {finetune_metrics['Precision']:<12.4f} "
              f"{finetune_metrics['Recall']:<8.4f} {finetune_metrics['F1']:<8.4f}")
    
    # 域适应效果分析
    if baseline_metrics is not None and dann_metrics is not None:
        adaptation_gain = dann_metrics['AUC'] - baseline_metrics['AUC']
        print(f"\n域适应效果 (DANN-Baseline): {adaptation_gain:.4f}")
        
        if adaptation_gain > 0:
            print("✅ 域适应有效！DANN在目标域上性能优于Baseline")
        else:
            print("❌ 域适应效果不明显，DANN在目标域上性能未超过Baseline")


def main():
    parser = argparse.ArgumentParser(description='改进的欺诈检测域适应项目主脚本')
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
    parser.add_argument('--evaluate_on_target', action='store_true',
                       help='是否在目标域上评估所有模型')
    
    args = parser.parse_args()
    
    # 设备配置
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("="*60)
    print("欺诈检测域适应项目（改进版）")
    print("="*60)
    print(f"设备: {device}")
    print(f"训练模式: {args.mode}")
    print(f"在目标域上评估: {args.evaluate_on_target}")
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
    
    baseline_target_metrics = None
    dann_target_metrics = None
    finetune_target_metrics = None
    
    if args.mode in ['baseline', 'all']:
        baseline_model, _, _, baseline_target_metrics = train_and_evaluate_baseline(
            source_features, source_labels, target_features, target_labels, device, config, args.evaluate_on_target
        )
    
    if args.mode in ['cdat_fd', 'all']:
        if target_features is not None:
            dann_model, _, _, dann_target_metrics = train_and_evaluate_dann(
                source_features, source_labels, target_features, target_labels, device, config, args.evaluate_on_target
            )
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
            finetune_model, _, _, finetune_target_metrics = train_and_evaluate_finetune(
                source_model, source_features, target_features, target_labels, device, config
            )
    
    # 性能对比
    if args.evaluate_on_target:
        compare_models(baseline_target_metrics, dann_target_metrics, finetune_target_metrics)
    
    # 可视化
    if args.visualize and baseline_model is not None and dann_model is not None and target_features is not None:
        # 对特征进行对齐，确保源域和目标域特征维度相同
        source_features_aligned, target_features_aligned, _ = align_features(
            source_features, target_features, method='pca'
        )
        
        # 创建用于可视化的DataLoader
        source_vis_loader = create_dataloader(
            source_features_aligned[:2000], source_labels[:2000],
            batch_size=args.batch_size, shuffle=False
        )
        target_vis_loader = create_dataloader(
            target_features_aligned[:2000], None,
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