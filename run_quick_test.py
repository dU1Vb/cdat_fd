"""
快速测试脚本
用于快速验证所有模块是否正常工作
"""
import torch
import numpy as np
from pathlib import Path

print("="*60)
print("快速测试 - 验证所有模块")
print("="*60)

# 测试1: 导入所有模块
print("\n[1] 测试模块导入...")
try:
    from models.baseline import BaselineFraudModel
    from models.cdat_fd import DANN
    from models.finetune import FineTuneModel
    from trainers.trainer_baseline import BaselineTrainer
    from trainers.trainer_cdat_fd import CDATFDTrainer
    from trainers.trainer_finetune import FineTuneTrainer
    from evaluation.evaluation import evaluate_fraud_model
    from visualization.tsne_plot import extract_features, plot_tsne
    from data import load_source_data, create_dataloader
    print("[OK] 所有模块导入成功")
except Exception as e:
    print(f"[FAIL] 模块导入失败: {e}")
    exit(1)

# 测试2: 创建模型
print("\n[2] 测试模型创建...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  使用设备: {device}")
    
    baseline = BaselineFraudModel(input_dim=30)
    dann = DANN(src_input_dim=30, tgt_input_dim=30)
    finetune = FineTuneModel(input_dim=30)
    print("[OK] 所有模型创建成功")
except Exception as e:
    print(f"[FAIL] 模型创建失败: {e}")
    exit(1)

# 测试3: 创建模拟数据
print("\n[3] 测试数据加载...")
try:
    # 创建模拟数据
    n_samples = 100
    n_features = 30
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples).astype(float)
    
    # 创建DataLoader
    loader = create_dataloader(X, y, batch_size=32, shuffle=True)
    print(f"[OK] 数据加载器创建成功: {len(loader)} batches")
except Exception as e:
    print(f"[FAIL] 数据加载失败: {e}")
    exit(1)

# 测试4: 前向传播
print("\n[4] 测试模型前向传播...")
try:
    baseline.eval()
    dann.eval()
    finetune.eval()
    
    x = torch.randn(10, 30).to(device)
    
    with torch.no_grad():
        out1 = baseline(x.to(device))
        out2, _ = dann(x.to(device), lambda_=0.5)
        out3 = finetune(x.to(device))
    
    print(f"[OK] Baseline输出形状: {out1.shape}")
    print(f"[OK] DANN输出形状: {out2.shape}")
    print(f"[OK] FineTune输出形状: {out3.shape}")
except Exception as e:
    print(f"[FAIL] 前向传播失败: {e}")
    exit(1)

# 测试5: 训练器创建
print("\n[5] 测试训练器创建...")
try:
    baseline_trainer = BaselineTrainer(baseline, loader, device=device, lr=1e-3)
    dann_trainer = CDATFDTrainer(dann, loader, loader, device=device, lr=1e-3)
    finetune_trainer = FineTuneTrainer(finetune, loader, device=device, lr=1e-4)
    print("[OK] 所有训练器创建成功")
except Exception as e:
    print(f"[FAIL] 训练器创建失败: {e}")
    exit(1)

# 测试6: 评估函数
print("\n[6] 测试评估函数...")
try:
    metrics = evaluate_fraud_model(baseline, loader, device=device)
    print(f"[OK] 评估成功: AUC={metrics['AUC']:.4f}")
except Exception as e:
    print(f"[FAIL] 评估失败: {e}")
    exit(1)

# 测试7: 检查数据文件
print("\n[7] 检查数据文件...")
source_path = Path("data/creditcard/creditcard.csv")
target_path = Path("data/ieee_fraud/train_transaction.csv")

if source_path.exists():
    print(f"[OK] 源域数据文件存在: {source_path}")
else:
    print(f"[WARN] 源域数据文件不存在: {source_path}")

if target_path.exists():
    print(f"[OK] 目标域数据文件存在: {target_path}")
else:
    print(f"[WARN] 目标域数据文件不存在: {target_path}")

print("\n" + "="*60)
print("快速测试完成！")
print("="*60)
print("\n如果所有测试通过，可以运行主脚本:")
print("  python main.py --mode all --epochs 5")
print("="*60)

