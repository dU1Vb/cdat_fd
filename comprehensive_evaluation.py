"""
改进的模型评估脚本
在源域和目标域上全面评估所有模型
"""
import torch
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models.baseline import BaselineFraudModel
from models.cdat_fd import DANN
from models.finetune import FineTuneModel
from data.loaders import load_source_data, load_target_train_data, load_target_test_data, create_dataloader
from data.preprocessing import preprocess_features, align_features
from evaluation.evaluation import evaluate_fraud_model, print_metrics


def evaluate_on_domain(model, loader, domain_name, device, find_optimal_threshold=True):
    """在指定域上评估模型"""
    print(f"\n[{domain_name}] 评估结果:")
    metrics = evaluate_fraud_model(model, loader, device=device, find_optimal_threshold=find_optimal_threshold)
    print_metrics(metrics, prefix="  ", verbose=True)
    return metrics


def comprehensive_evaluation(args):
    """全面的模型评估"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载源域数据
    print("\n[1] 加载源域数据...")
    source_features, source_labels = load_source_data(args.source_data)
    print(f"源域数据: {source_features.shape[0]} 样本, {source_features.shape[1]} 特征")
    
    # 加载目标域数据
    print("\n[2] 加载目标域数据...")
    try:
        target_train_features, target_train_labels = load_target_train_data(
            args.target_train_trans,
            args.target_train_id if Path(args.target_train_id).exists() else None
        )
        print(f"目标域训练数据: {target_train_features.shape[0]} 样本, {target_train_features.shape[1]} 特征")
        has_target_labels = True
    except Exception as e:
        print(f"目标域训练数据加载失败: {e}")
        # 尝试加载测试数据作为无标签数据
        target_train_features, _ = load_target_test_data(
            args.target_train_trans,
            args.target_train_id if Path(args.target_train_id).exists() else None
        )
        target_train_labels = np.zeros(len(target_train_features))  # 伪标签
        print(f"使用测试数据作为目标域（无标签）: {target_train_features.shape[0]} 样本")
        has_target_labels = False
    
    # 加载目标域测试数据（如果有）
    target_test_features = None
    target_test_labels = None
    if args.target_test_trans and Path(args.target_test_trans).exists():
        try:
            target_test_features, target_test_labels = load_target_test_data(
                args.target_test_trans,
                args.target_test_id if Path(args.target_test_id).exists() else None
            )
            print(f"目标域测试数据: {target_test_features.shape[0]} 样本, {target_test_features.shape[1]} 特征")
        except Exception as e:
            print(f"目标域测试数据加载失败: {e}")
    
    # 数据预处理和特征对齐
    print("\n[3] 数据预处理...")
    
    # 源域预处理
    source_features_processed, source_scaler = preprocess_features(source_features, fit=True)
    
    # 目标域预处理（使用源域的scaler）
    target_train_features_processed, _ = preprocess_features(
        target_train_features, scaler=source_scaler, fit=False
    )
    
    if target_test_features is not None:
        target_test_features_processed, _ = preprocess_features(
            target_test_features, scaler=source_scaler, fit=False
        )
    
    # 特征对齐
    source_features_aligned, target_train_features_aligned, input_dim = align_features(
        source_features_processed, target_train_features_processed, method='pca'
    )
    
    if target_test_features is not None:
        _, target_test_features_aligned, _ = align_features(
            source_features_processed, target_test_features_processed, method='pca'
        )
    else:
        target_test_features_aligned = None
    
    # 创建数据加载器
    print("\n[4] 创建数据加载器...")
    
    # 源域数据加载器
    source_train_loader = create_dataloader(
        source_features_aligned, source_labels,
        batch_size=args.batch_size, shuffle=True
    )
    source_val_loader = create_dataloader(
        source_features_aligned, source_labels,
        batch_size=args.batch_size, shuffle=False
    )
    
    # 目标域训练数据加载器
    target_train_loader = create_dataloader(
        target_train_features_aligned, target_train_labels,
        batch_size=args.batch_size, shuffle=True
    )
    target_train_val_loader = create_dataloader(
        target_train_features_aligned, target_train_labels,
        batch_size=args.batch_size, shuffle=False
    )
    
    # 目标域测试数据加载器（如果有）
    target_test_loader = None
    if target_test_features_aligned is not None:
        target_test_loader = create_dataloader(
            target_test_features_aligned, target_test_labels,
            batch_size=args.batch_size, shuffle=False
        )
    
    # 加载已训练的模型
    print("\n[5] 加载已训练的模型...")
    
    # Baseline模型
    baseline_model = BaselineFraudModel(input_dim=input_dim)
    if Path(args.baseline_model_path).exists():
        baseline_model.load_state_dict(torch.load(args.baseline_model_path, map_location=device))
        baseline_model.to(device)
        print(f"[OK] Baseline模型加载成功: {args.baseline_model_path}")
    else:
        print(f"[WARN] Baseline模型文件不存在: {args.baseline_model_path}")
        baseline_model = None
    
    # DANN模型
    dann_model = DANN(src_input_dim=input_dim, tgt_input_dim=input_dim)
    if Path(args.dann_model_path).exists():
        dann_model.load_state_dict(torch.load(args.dann_model_path, map_location=device))
        dann_model.to(device)
        print(f"[OK] DANN模型加载成功: {args.dann_model_path}")
    else:
        print(f"[WARN] DANN模型文件不存在: {args.dann_model_path}")
        dann_model = None
    
    # Fine-tune模型
    finetune_model = FineTuneModel(input_dim=input_dim)
    if Path(args.finetune_model_path).exists():
        finetune_model.load_state_dict(torch.load(args.finetune_model_path, map_location=device))
        finetune_model.to(device)
        print(f"[OK] Fine-tune模型加载成功: {args.finetune_model_path}")
    else:
        print(f"[WARN] Fine-tune模型文件不存在: {args.finetune_model_path}")
        finetune_model = None
    
    # 全面评估
    print("\n" + "="*60)
    print("全面模型评估")
    print("="*60)
    
    results = {}
    
    # 评估Baseline模型
    if baseline_model is not None:
        print("\n" + "-"*40)
        print("Baseline模型评估")
        print("-"*40)
        
        # 在源域上评估
        results['baseline_source'] = evaluate_on_domain(
            baseline_model, source_val_loader, "源域", device
        )
        
        # 在目标域上评估
        results['baseline_target_train'] = evaluate_on_domain(
            baseline_model, target_train_val_loader, "目标域(训练)", device
        )
        
        if target_test_loader is not None:
            results['baseline_target_test'] = evaluate_on_domain(
                baseline_model, target_test_loader, "目标域(测试)", device
            )
    
    # 评估DANN模型
    if dann_model is not None:
        print("\n" + "-"*40)
        print("DANN模型评估")
        print("-"*40)
        
        # 在源域上评估
        results['dann_source'] = evaluate_on_domain(
            dann_model, source_val_loader, "源域", device
        )
        
        # 在目标域上评估
        results['dann_target_train'] = evaluate_on_domain(
            dann_model, target_train_val_loader, "目标域(训练)", device
        )
        
        if target_test_loader is not None:
            results['dann_target_test'] = evaluate_on_domain(
                dann_model, target_test_loader, "目标域(测试)", device
            )
    
    # 评估Fine-tune模型
    if finetune_model is not None:
        print("\n" + "-"*40)
        print("Fine-tune模型评估")
        print("-"*40)
        
        # 在源域上评估
        results['finetune_source'] = evaluate_on_domain(
            finetune_model, source_val_loader, "源域", device
        )
        
        # 在目标域上评估
        results['finetune_target_train'] = evaluate_on_domain(
            finetune_model, target_train_val_loader, "目标域(训练)", device
        )
        
        if target_test_loader is not None:
            results['finetune_target_test'] = evaluate_on_domain(
                finetune_model, target_test_loader, "目标域(测试)", device
            )
    
    # 性能对比分析
    print("\n" + "="*60)
    print("性能对比分析")
    print("="*60)
    
    # 目标域性能对比
    print("\n[目标域性能对比]")
    print("-" * 40)
    
    target_metrics = ['AUC', 'Precision', 'Recall', 'F1']
    models = []
    
    if baseline_model is not None:
        models.append(('Baseline', results['baseline_target_train']))
    if dann_model is not None:
        models.append(('DANN', results['dann_target_train']))
    if finetune_model is not None:
        models.append(('Fine-tune', results['finetune_target_train']))
    
    # 打印对比表格
    print(f"{'模型':<12} {'AUC':<8} {'Precision':<12} {'Recall':<8} {'F1':<8}")
    print("-" * 50)
    
    for model_name, metrics in models:
        print(f"{model_name:<12} {metrics['AUC']:<8.4f} {metrics['Precision']:<12.4f} "
              f"{metrics['Recall']:<8.4f} {metrics['F1']:<8.4f}")
    
    # 域适应效果分析
    print("\n[域适应效果分析]")
    print("-" * 40)
    
    if baseline_model is not None and dann_model is not None:
        source_auc = results['baseline_source']['AUC']
        target_auc = results['baseline_target_train']['AUC']
        dann_target_auc = results['dann_target_train']['AUC']
        
        print(f"Baseline 源域AUC: {source_auc:.4f}")
        print(f"Baseline 目标域AUC: {target_auc:.4f}")
        print(f"DANN 目标域AUC: {dann_target_auc:.4f}")
        
        # 计算域差异和域适应效果
        domain_gap = source_auc - target_auc
        adaptation_gain = dann_target_auc - target_auc
        
        print(f"\n域差异 (源域-目标域): {domain_gap:.4f}")
        print(f"域适应效果 (DANN-Baseline): {adaptation_gain:.4f}")
        
        if adaptation_gain > 0:
            print("✅ 域适应有效！DANN在目标域上性能优于Baseline")
        else:
            print("❌ 域适应效果不明显，DANN在目标域上性能未超过Baseline")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='全面模型评估脚本')
    parser.add_argument('--source_data', type=str,
                       default='data/creditcard/creditcard.csv',
                       help='源域数据路径')
    parser.add_argument('--target_train_trans', type=str,
                       default='data/ieee_fraud/train_transaction.csv',
                       help='目标域训练交易数据路径')
    parser.add_argument('--target_train_id', type=str,
                       default='data/ieee_fraud/train_identity.csv',
                       help='目标域训练身份数据路径')
    parser.add_argument('--target_test_trans', type=str,
                       default='data/ieee_fraud/test_transaction.csv',
                       help='目标域测试交易数据路径')
    parser.add_argument('--target_test_id', type=str,
                       default='data/ieee_fraud/test_identity.csv',
                       help='目标域测试身份数据路径')
    parser.add_argument('--baseline_model_path', type=str,
                       default='checkpoints/baseline_model.pth',
                       help='Baseline模型路径')
    parser.add_argument('--dann_model_path', type=str,
                       default='checkpoints/dann_model.pth',
                       help='DANN模型路径')
    parser.add_argument('--finetune_model_path', type=str,
                       default='checkpoints/finetune_model.pth',
                       help='Fine-tune模型路径')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    
    args = parser.parse_args()
    
    comprehensive_evaluation(args)