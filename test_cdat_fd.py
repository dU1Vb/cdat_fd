"""测试CDAT-FD.py是否能正常运行"""
import torch
import sys

# 导入模型
try:
    # 注意：Python模块名不能有连字符，需要重命名文件或使用importlib
    import importlib.util
    spec = importlib.util.spec_from_file_location("cdat_fd", "CDAT-FD.py")
    cdat_fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cdat_fd)
    DANN = cdat_fd.DANN
    FeatureEncoder = cdat_fd.FeatureEncoder
    FraudClassifier = cdat_fd.FraudClassifier
    DomainDiscriminator = cdat_fd.DomainDiscriminator
    grl = cdat_fd.grl
    print("[OK] 成功导入所有模块")
except ImportError as e:
    print(f"[FAIL] 导入失败: {e}")
    sys.exit(1)

# 测试1: 创建模型实例
try:
    model = DANN(src_input_dim=100, tgt_input_dim=100)
    print("[OK] 成功创建DANN模型实例")
except Exception as e:
    print(f"[FAIL] 创建模型失败: {e}")
    sys.exit(1)

# 测试2: 前向传播（正常情况）
try:
    x = torch.randn(32, 100)  # batch_size=32, input_dim=100
    class_pred, domain_pred = model(x)
    print(f"[OK] 前向传播成功")
    print(f"  - 分类输出形状: {class_pred.shape}")
    print(f"  - 域判别输出形状: {domain_pred.shape}")
except Exception as e:
    print(f"[FAIL] 前向传播失败: {e}")
    sys.exit(1)

# 测试3: 测试不同输入维度的情况
try:
    model2 = DANN(src_input_dim=50, tgt_input_dim=100)
    x_small = torch.randn(32, 50)
    # 这应该会失败，因为encoder期望100维输入
    try:
        class_pred, domain_pred = model2(x_small)
        print("[WARN] 警告: 不同维度输入应该失败但没有失败")
    except RuntimeError as e:
        print(f"[OK] 正确捕获了维度不匹配错误: {type(e).__name__}")
except Exception as e:
    print(f"[FAIL] 测试维度不匹配时出错: {e}")

# 测试4: 测试batch_size=1的情况（BatchNorm问题）
try:
    x_single = torch.randn(1, 100)  # batch_size=1
    class_pred, domain_pred = model(x_single)
    print("[OK] batch_size=1时也能正常工作")
except Exception as e:
    print(f"[WARN] batch_size=1时出错: {e}")
    print("  建议: 在训练时确保batch_size > 1，或在BatchNorm中使用track_running_stats=False")

# 测试5: 测试梯度反转层
try:
    x = torch.randn(32, 100, requires_grad=True)
    features = model.encoder(x)
    reversed_features = grl(features, lambda_=0.5)
    print("[OK] 梯度反转层工作正常")
except Exception as e:
    print(f"[FAIL] 梯度反转层测试失败: {e}")

print("\n" + "="*50)
print("测试完成！代码基本可以运行，但需要注意上述问题。")

