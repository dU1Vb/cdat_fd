"""测试改进后的模型是否能正常导入和实例化"""
import torch
from models.cdat_fd import DANN, FeatureEncoder, FraudClassifier, DomainDiscriminator
from models.baseline import BaselineFraudModel

print("=" * 60)
print("测试改进后的模型")
print("=" * 60)

# 测试FeatureEncoder
print("\n[1] 测试FeatureEncoder...")
try:
    encoder = FeatureEncoder(input_dim=30, hidden_dims=[256, 128, 128, 64], output_dim=64)
    x = torch.randn(32, 30)
    out = encoder(x)
    print(f"  ✅ FeatureEncoder测试成功: {x.shape} -> {out.shape}")
except Exception as e:
    print(f"  ❌ FeatureEncoder测试失败: {e}")

# 测试FraudClassifier
print("\n[2] 测试FraudClassifier...")
try:
    classifier = FraudClassifier(input_dim=64, hidden_dims=[64, 32])
    x = torch.randn(32, 64)
    out = classifier(x)
    print(f"  ✅ FraudClassifier测试成功: {x.shape} -> {out.shape}")
except Exception as e:
    print(f"  ❌ FraudClassifier测试失败: {e}")

# 测试DomainDiscriminator
print("\n[3] 测试DomainDiscriminator...")
try:
    discriminator = DomainDiscriminator(input_dim=64, hidden_dims=[128, 64, 32])
    x = torch.randn(32, 64)
    out = discriminator(x)
    print(f"  ✅ DomainDiscriminator测试成功: {x.shape} -> {out.shape}")
except Exception as e:
    print(f"  ❌ DomainDiscriminator测试失败: {e}")

# 测试DANN
print("\n[4] 测试DANN模型...")
try:
    dann = DANN(src_input_dim=30, tgt_input_dim=30)
    x = torch.randn(32, 30)
    class_pred, domain_pred = dann(x, lambda_=1.0)
    print(f"  ✅ DANN模型测试成功:")
    print(f"     输入: {x.shape}")
    print(f"     分类输出: {class_pred.shape}")
    print(f"     域判别输出: {domain_pred.shape}")
except Exception as e:
    print(f"  ❌ DANN模型测试失败: {e}")

# 测试Baseline模型
print("\n[5] 测试Baseline模型...")
try:
    baseline = BaselineFraudModel(input_dim=30)
    x = torch.randn(32, 30)
    out = baseline(x)
    print(f"  ✅ Baseline模型测试成功: {x.shape} -> {out.shape}")
except Exception as e:
    print(f"  ❌ Baseline模型测试失败: {e}")

# 统计参数量
print("\n" + "=" * 60)
print("模型参数量统计")
print("=" * 60)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

encoder = FeatureEncoder(input_dim=30, hidden_dims=[256, 128, 128, 64], output_dim=64)
classifier = FraudClassifier(input_dim=64, hidden_dims=[64, 32])
discriminator = DomainDiscriminator(input_dim=64, hidden_dims=[128, 64, 32])
dann = DANN(src_input_dim=30, tgt_input_dim=30)
baseline = BaselineFraudModel(input_dim=30)

print(f"\nFeatureEncoder参数量: {count_parameters(encoder):,}")
print(f"FraudClassifier参数量: {count_parameters(classifier):,}")
print(f"DomainDiscriminator参数量: {count_parameters(discriminator):,}")
print(f"DANN总参数量: {count_parameters(dann):,}")
print(f"Baseline参数量: {count_parameters(baseline):,}")

print("\n" + "=" * 60)
print("✅ 所有测试完成！")
print("=" * 60)

