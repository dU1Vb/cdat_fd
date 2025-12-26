# 数据集组织建议

## 📁 推荐文件夹结构

### 方案一：按域组织（推荐）
```
data/
├── source_domain/          # 源域数据
│   └── creditcard/
│       └── creditcard.csv
└── target_domain/          # 目标域数据
    └── ieee_fraud/
        ├── train_transaction.csv
        ├── train_identity.csv
        ├── test_transaction.csv
        ├── test_identity.csv
        └── sample_submission.csv
```

### 方案二：统一数据集文件夹（更简洁）
```
data/
├── source/                 # 源域
│   └── creditcard.csv
└── target/                 # 目标域
    ├── train_transaction.csv
    ├── train_identity.csv
    ├── test_transaction.csv
    ├── test_identity.csv
    └── sample_submission.csv
```

### 方案三：按数据集名称组织（最清晰）
```
data/
├── creditcard/             # 源域：信用卡欺诈数据集
│   └── creditcard.csv
└── ieee_fraud/            # 目标域：IEEE欺诈检测数据集
    ├── train_transaction.csv
    ├── train_identity.csv
    ├── test_transaction.csv
    ├── test_identity.csv
    └── sample_submission.csv
```

## 🎯 推荐方案：方案三（按数据集名称组织）

**理由：**
1. ✅ 清晰明确：一眼就能看出是哪个数据集
2. ✅ 易于扩展：未来添加新数据集时结构清晰
3. ✅ 符合常见实践：大多数ML项目都采用这种结构
4. ✅ 便于数据加载：路径逻辑清晰

## 📝 文件命名规范

### 源域文件
- `creditcard.csv` ✅ (保持原样，简洁明了)

### 目标域文件
建议统一命名风格：

**当前命名** → **推荐命名**
- `train_transaction.csv` ✅ (保持)
- `train_identity.csv` ✅ (保持)
- `test_transaction.csv` ✅ (保持)
- `test_identity.csv` ✅ (保持)
- `sample_submission.csv` ✅ (保持)

**或者更明确的命名：**
- `ieee_train_transaction.csv`
- `ieee_train_identity.csv`
- `ieee_test_transaction.csv`
- `ieee_test_identity.csv`
- `ieee_sample_submission.csv`

## 🔧 实施步骤

### 1. 创建新的数据结构
```bash
# 创建主数据文件夹
mkdir data

# 创建源域文件夹
mkdir data/creditcard

# 创建目标域文件夹
mkdir data/ieee_fraud

# 移动文件
mv archive/creditcard.csv data/creditcard/
mv ieee-fraud-detection/* data/ieee_fraud/
```

### 2. 更新后的项目结构
```
Graduation project/
├── data/                   # 统一数据文件夹
│   ├── creditcard/        # 源域
│   │   └── creditcard.csv
│   └── ieee_fraud/        # 目标域
│       ├── train_transaction.csv
│       ├── train_identity.csv
│       ├── test_transaction.csv
│       ├── test_identity.csv
│       └── sample_submission.csv
├── models/
├── trainers/
├── evaluation/
├── visualization/
└── ...
```

## 📚 数据加载模块建议

建议创建 `data/` 或 `datasets/` 模块来统一管理数据加载：

```
data/
├── __init__.py
├── loaders.py          # 数据加载器
└── preprocess.py       # 数据预处理
```

## 🎨 命名约定总结

| 类型 | 命名规范 | 示例 |
|------|---------|------|
| 主数据文件夹 | `data/` | `data/` |
| 源域文件夹 | 数据集名称 | `creditcard/` |
| 目标域文件夹 | 数据集名称 | `ieee_fraud/` |
| CSV文件 | 小写+下划线 | `train_transaction.csv` |
| 训练数据 | `train_*.csv` | `train_transaction.csv` |
| 测试数据 | `test_*.csv` | `test_transaction.csv` |

## ✅ 最终推荐结构

```
data/
├── creditcard/                    # 源域数据集
│   └── creditcard.csv
└── ieee_fraud/                    # 目标域数据集
    ├── train_transaction.csv      # 训练集-交易数据
    ├── train_identity.csv         # 训练集-身份数据
    ├── test_transaction.csv       # 测试集-交易数据
    ├── test_identity.csv          # 测试集-身份数据
    └── sample_submission.csv      # 提交样例
```

**优势：**
- ✅ 结构清晰，一目了然
- ✅ 符合域适应项目的逻辑
- ✅ 易于编写数据加载代码
- ✅ 便于版本控制和文档管理

