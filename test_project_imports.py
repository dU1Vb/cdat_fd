"""
测试项目所有模块的导入是否正常
"""
import sys
from pathlib import Path

print("=" * 60)
print("项目模块导入测试")
print("=" * 60)

errors = []
success = []

# 测试模型模块
print("\n[1] 测试模型模块...")
try:
    from models.baseline import BaselineFraudModel
    success.append("models.baseline")
    print("  [OK] models.baseline")
except Exception as e:
    errors.append(("models.baseline", str(e)))
    print(f"  [FAIL] models.baseline: {e}")

try:
    from models.cdat_fd import DANN, FeatureEncoder, FraudClassifier, DomainDiscriminator, grl
    success.append("models.cdat_fd")
    print("  [OK] models.cdat_fd")
except Exception as e:
    errors.append(("models.cdat_fd", str(e)))
    print(f"  [FAIL] models.cdat_fd: {e}")

try:
    from models.finetune import FineTuneModel
    success.append("models.finetune")
    print("  [OK] models.finetune")
except Exception as e:
    errors.append(("models.finetune", str(e)))
    print(f"  [FAIL] models.finetune: {e}")

# 测试训练器模块
print("\n[2] 测试训练器模块...")
try:
    from trainers.trainer_baseline import BaselineTrainer
    success.append("trainers.trainer_baseline")
    print("  [OK] trainers.trainer_baseline")
except Exception as e:
    errors.append(("trainers.trainer_baseline", str(e)))
    print(f"  [FAIL] trainers.trainer_baseline: {e}")

try:
    from trainers.trainer_cdat_fd import CDATFDTrainer, lambda_schedule
    success.append("trainers.trainer_cdat_fd")
    print("  [OK] trainers.trainer_cdat_fd")
except Exception as e:
    errors.append(("trainers.trainer_cdat_fd", str(e)))
    print(f"  [FAIL] trainers.trainer_cdat_fd: {e}")

try:
    from trainers.trainer_finetune import FineTuneTrainer
    success.append("trainers.trainer_finetune")
    print("  [OK] trainers.trainer_finetune")
except Exception as e:
    errors.append(("trainers.trainer_finetune", str(e)))
    print(f"  [FAIL] trainers.trainer_finetune: {e}")

# 测试评估模块
print("\n[3] 测试评估模块...")
try:
    from evaluation.evaluation import evaluate_fraud_model, print_metrics
    success.append("evaluation.evaluation")
    print("  [OK] evaluation.evaluation")
except Exception as e:
    errors.append(("evaluation.evaluation", str(e)))
    print(f"  [FAIL] evaluation.evaluation: {e}")

# 测试可视化模块
print("\n[4] 测试可视化模块...")
try:
    from visualization.tsne_plot import extract_features, plot_tsne
    success.append("visualization.tsne_plot")
    print("  [OK] visualization.tsne_plot")
except Exception as e:
    errors.append(("visualization.tsne_plot", str(e)))
    print(f"  [FAIL] visualization.tsne_plot: {e}")

# 测试模型实例化
print("\n[5] 测试模型实例化...")
try:
    import torch
    model1 = BaselineFraudModel(input_dim=30)
    print("  [OK] BaselineFraudModel 实例化成功")
    
    model2 = DANN(src_input_dim=30, tgt_input_dim=30)
    print("  [OK] DANN 实例化成功")
    
    model3 = FineTuneModel(input_dim=30)
    print("  [OK] FineTuneModel 实例化成功")
except Exception as e:
    errors.append(("模型实例化", str(e)))
    print(f"  [FAIL] 模型实例化失败: {e}")

# 总结
print("\n" + "=" * 60)
print("测试总结")
print("=" * 60)
print(f"成功: {len(success)}/{len(success) + len(errors)}")
print(f"失败: {len(errors)}/{len(success) + len(errors)}")

if errors:
    print("\n错误详情:")
    for module, error in errors:
        print(f"  - {module}: {error}")
else:
    print("\n[OK] 所有模块导入成功！项目代码可以正常运行。")

print("=" * 60)

