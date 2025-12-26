# 欺诈检测域适应项目

基于域适应的欺诈检测系统，支持从源域（Credit Card）到目标域（IEEE Fraud Detection）的知识迁移。

## 📁 项目结构

```
Graduation project/
├── data/                    # 数据文件夹
│   ├── creditcard/         # 源域数据
│   └── ieee_fraud/         # 目标域数据
├── models/                  # 模型定义
│   ├── baseline.py        # Baseline模型
│   ├── cdat_fd.py         # DANN域适应模型
│   └── finetune.py        # FineTune模型
├── trainers/                # 训练器
│   ├── trainer_baseline.py
│   ├── trainer_cdat_fd.py
│   └── trainer_finetune.py
├── evaluation/              # 评估模块
│   └── evaluation.py
├── visualization/           # 可视化模块
│   └── tsne_plot.py
├── main.py                  # 主运行脚本
└── README.md
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 激活conda环境
conda activate ms_env

# 安装依赖（如果还没有）
pip install torch pandas numpy scikit-learn matplotlib
```

### 2. 数据准备

确保数据文件在以下位置：
- 源域: `data/creditcard/creditcard.csv`
- 目标域: `data/ieee_fraud/train_transaction.csv` 和 `data/ieee_fraud/train_identity.csv`

### 3. 运行主脚本

#### 训练所有模型（推荐）
```bash
python main.py --mode all --epochs 20 --batch_size 64
```

#### 只训练Baseline模型
```bash
python main.py --mode baseline --epochs 20
```

#### 只训练域适应模型（CDAT-FD）
```bash
python main.py --mode cdat_fd --epochs 50
```

#### 训练并生成可视化
```bash
python main.py --mode all --epochs 20 --visualize
```

## 📋 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 训练模式: `baseline`, `cdat_fd`, `finetune`, `all` | `all` |
| `--source_data` | 源域数据路径 | `data/creditcard/creditcard.csv` |
| `--target_train_trans` | 目标域训练交易数据 | `data/ieee_fraud/train_transaction.csv` |
| `--target_train_id` | 目标域训练身份数据 | `data/ieee_fraud/train_identity.csv` |
| `--epochs` | 训练轮数 | `20` |
| `--finetune_epochs` | 微调轮数 | `10` |
| `--batch_size` | 批次大小 | `64` |
| `--lr` | 学习率 | `1e-3` |
| `--device` | 计算设备: `auto`, `cuda`, `cpu` | `auto` |
| `--save_dir` | 模型保存目录 | `checkpoints` |
| `--visualize` | 是否生成t-SNE可视化 | `False` |
| `--vis_dir` | 可视化结果保存目录 | `visualizations` |

## 💡 使用示例

### 示例1: 完整训练流程
```bash
# 训练所有模型，生成可视化，保存模型
python main.py \
    --mode all \
    --epochs 30 \
    --batch_size 128 \
    --lr 0.001 \
    --visualize \
    --save_dir checkpoints \
    --vis_dir visualizations
```

### 示例2: 快速测试（少量epoch）
```bash
# 快速测试Baseline模型
python main.py --mode baseline --epochs 5 --batch_size 32
```

### 示例3: 使用GPU训练
```bash
# 明确指定使用CUDA
python main.py --mode all --device cuda --epochs 50
```

### 示例4: 自定义数据路径
```bash
python main.py \
    --source_data data/creditcard/creditcard.csv \
    --target_train_trans data/ieee_fraud/train_transaction.csv \
    --target_train_id data/ieee_fraud/train_identity.csv
```

## 📊 模型说明

### 1. Baseline模型
- **描述**: 仅在源域上训练的基础模型
- **用途**: 作为对比基准
- **输入**: 源域特征
- **输出**: 欺诈概率

### 2. CDAT-FD模型（DANN）
- **描述**: 域对抗神经网络，实现域适应
- **用途**: 从源域迁移知识到目标域
- **特点**: 同时优化分类损失和域对抗损失

### 3. FineTune模型
- **描述**: 在目标域上微调的模型
- **用途**: 进一步优化目标域性能
- **特点**: 使用预训练的encoder，只训练分类器

## 📈 输出说明

### 训练过程
- 每个epoch会显示训练损失
- 训练结束后会显示验证集评估指标（AUC, Precision, Recall, F1, KS）

### 模型保存
- 训练好的模型会保存到 `checkpoints/` 目录
- 文件命名: `baseline_model.pth`, `dann_model.pth`, `finetune_model.pth`

### 可视化
- 如果使用 `--visualize` 参数，会生成t-SNE可视化图
- 保存到 `visualizations/` 目录
- 包含Baseline和DANN模型的特征分布对比

## 🔧 代码模块使用

### 单独使用数据加载模块
```python
from data import load_source_dataloader, load_target_train_dataloader

# 加载源域数据
source_loader = load_source_dataloader("data/creditcard/creditcard.csv", batch_size=64)

# 加载目标域数据
target_loader = load_target_train_dataloader(
    "data/ieee_fraud/train_transaction.csv",
    "data/ieee_fraud/train_identity.csv",
    batch_size=64
)
```

### 单独使用模型
```python
from models.baseline import BaselineFraudModel
from models.cdat_fd import DANN

# 创建模型
baseline = BaselineFraudModel(input_dim=30)
dann = DANN(src_input_dim=30, tgt_input_dim=30)
```

### 单独使用评估模块
```python
from evaluation import evaluate_fraud_model, print_metrics

# 评估模型
metrics = evaluate_fraud_model(model, dataloader, device="cuda")
print_metrics(metrics)
```

## ⚠️ 注意事项

1. **数据路径**: 确保数据文件路径正确
2. **内存**: IEEE数据集较大，注意内存使用
3. **GPU**: 如果有GPU，建议使用 `--device cuda` 加速训练
4. **批次大小**: 根据内存情况调整 `--batch_size`
5. **特征维度**: 源域和目标域特征维度会自动对齐

## 🐛 常见问题

### Q: 数据加载失败？
A: 检查数据文件路径是否正确，确保CSV文件存在

### Q: 内存不足？
A: 减小 `--batch_size` 或使用数据采样

### Q: 训练很慢？
A: 使用GPU (`--device cuda`) 或减小数据量

### Q: 如何只评估不训练？
A: 可以单独使用评估模块，加载已保存的模型

## 📝 许可证

本项目用于毕业设计研究。

