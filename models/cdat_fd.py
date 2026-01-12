import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import math

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grl(x, lambda_=1.0):
    return GradientReversal.apply(x, lambda_)


class ResidualBlock(nn.Module):
    """残差块，用于构建深层网络"""
    def __init__(self, dim, dropout=0.2, use_batch_norm=True):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.use_batch_norm = use_batch_norm
        
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(dim, track_running_stats=True)
            self.bn2 = nn.BatchNorm1d(dim, track_running_stats=True)
        else:
            self.bn1 = nn.LayerNorm(dim)
            self.bn2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        residual = x
        
        # 第一个全连接层
        out = self.fc1(x)
        if self.use_batch_norm and x.size(0) > 1:
            out = self.bn1(out)
        elif not self.use_batch_norm:
            out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # 第二个全连接层
        out = self.fc2(out)
        if self.use_batch_norm and x.size(0) > 1:
            out = self.bn2(out)
        elif not self.use_batch_norm:
            out = self.bn2(out)
        
        # 残差连接
        out = out + residual
        out = F.relu(out)
        
        return out


class SelfAttention(nn.Module):
    """
    自注意力机制，用于捕获特征间的依赖关系
    适配2D输入 [batch_size, feature_dim]
    """
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim必须能被num_heads整除"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        # x shape: [batch_size, dim] 或 [batch_size, seq_len, dim]
        if x.dim() == 2:
            # 2D输入，添加序列维度
            x = x.unsqueeze(1)  # [batch_size, 1, dim]
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len, dim = x.size()
        residual = x
        
        # Layer Normalization
        x = self.layer_norm(x)
        
        # 计算Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.dim
        )
        
        # 输出投影
        output = self.output(attn_output)
        output = self.dropout(output)
        
        # 残差连接
        output = output + residual
        
        # 如果输入是2D，输出也应该是2D
        if squeeze_output:
            output = output.squeeze(1)  # [batch_size, dim]
        
        return output


class FeatureInteractionLayer(nn.Module):
    """特征交互层，用于捕获特征间的交互关系"""
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        
        # 特征交互
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        
        # 残差连接
        out = out + residual
        return out


class FeatureEncoder(nn.Module):
    """
    改进的特征编码器，包含：
    - 更深的网络结构（5-6层）
    - 残差连接
    - 自注意力机制
    - Dropout正则化
    - 特征交互层
    """
    def __init__(
        self, 
        input_dim, 
        hidden_dims=[256, 128, 128, 64], 
        output_dim=64,
        use_batch_norm=True,
        dropout=0.3,
        use_attention=True,
        use_residual=True,
        use_feature_interaction=True
    ):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_feature_interaction = use_feature_interaction
        
        # 第一层：输入层到第一个隐藏层
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0], track_running_stats=True) if use_batch_norm else nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 中间层：构建隐藏层和残差块
        self.hidden_layers = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            # 维度变化层
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1], track_running_stats=True) if use_batch_norm else nn.LayerNorm(hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            
            # 如果维度相同，添加残差块
            if use_residual and hidden_dims[i] == hidden_dims[i+1]:
                self.residual_blocks.append(ResidualBlock(hidden_dims[i], dropout=dropout, use_batch_norm=use_batch_norm))
        
        # 自注意力层（应用在第二个隐藏层后）
        if use_attention and len(hidden_dims) > 1:
            attention_dim = hidden_dims[1]
            self.attention = SelfAttention(attention_dim, num_heads=4, dropout=dropout)
        else:
            self.attention = None
        
        # 特征交互层（应用在倒数第二个隐藏层）
        if use_feature_interaction and len(hidden_dims) > 1:
            interaction_dim = hidden_dims[-1]  # 使用最后一个隐藏层的维度
            self.feature_interaction = FeatureInteractionLayer(
                interaction_dim, 
                hidden_dim=interaction_dim//2, 
                dropout=dropout
            )
        else:
            self.feature_interaction = None
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.BatchNorm1d(output_dim, track_running_stats=True) if use_batch_norm else nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)  # 输出层使用较小的dropout
        )
        
        self.hidden_dims = hidden_dims
    
    def forward(self, x):
        # 处理batch_size=1的情况
        if self.use_batch_norm and x.size(0) == 1:
            was_training = self.training
            if was_training:
                self.eval()
        
        # 第一层
        x = self.input_layer(x)
        
        # 中间层处理
        residual_idx = 0
        for i, hidden_layer in enumerate(self.hidden_layers):
            # 应用残差块（如果维度相同）
            if self.use_residual and residual_idx < len(self.residual_blocks):
                # 检查是否应该应用残差块
                if i < len(self.hidden_dims) - 1 and self.hidden_dims[i] == self.hidden_dims[i+1]:
                    x = self.residual_blocks[residual_idx](x)
                    residual_idx += 1
            
            x = hidden_layer(x)
            
            # 在第二个隐藏层后应用注意力
            if self.use_attention and self.attention is not None and i == 0:
                x = self.attention(x)
        
        # 特征交互层（在输出层之前）
        if self.use_feature_interaction and self.feature_interaction is not None:
            x = self.feature_interaction(x)
        
        # 输出层
        x = self.output_layer(x)
        
        if self.use_batch_norm and x.size(0) == 1 and was_training:
            self.train()
        
        return x

class FraudClassifier(nn.Module):
    """
    改进的欺诈分类器，使用更深的网络和正则化
    """
    def __init__(self, input_dim=64, hidden_dims=[64, 32], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # 处理batch_size=1的情况
        if x.size(0) == 1:
            was_training = self.training
            if was_training:
                self.eval()
            result = torch.sigmoid(self.net(x))
            if was_training:
                self.train()
            return result
        return torch.sigmoid(self.net(x))


class FinancialFeatureProcessor(nn.Module):
    """
    金融特征处理器
    专门处理金融领域的特征，添加金融风险相关的特征交互
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        # 金融特征交互层
        self.feature_interaction = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 风险因子提取
        self.risk_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 注意力机制，用于关注重要的金融特征
        self.attention = SelfAttention(hidden_dim // 2, num_heads=2, dropout=0.1)
        
        # 输出层，保持维度一致
        self.output_layer = nn.Linear(hidden_dim // 2, input_dim)
    
    def forward(self, x):
        # 金融特征交互
        x = self.feature_interaction(x)
        
        # 风险因子提取
        x = self.risk_extractor(x)
        
        # 注意力机制
        x = self.attention(x)
        
        # 恢复原始维度
        x = self.output_layer(x)
        
        return x


class DomainDiscriminator(nn.Module):
    """
    改进的域判别器，使用更深的网络和梯度惩罚
    """
    def __init__(self, input_dim=64, hidden_dims=[128, 64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),  # 使用LeakyReLU提高判别器性能
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层（2类：源域和目标域）
        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # 处理batch_size=1的情况
        if x.size(0) == 1:
            was_training = self.training
            if was_training:
                self.eval()
            result = self.net(x)
            if was_training:
                self.train()
            return result
        return self.net(x)

class DANCE(nn.Module):
    """
    基于对比学习的域适应模型（DANCE）
    结合了DANN的域对抗学习和对比学习机制，添加了金融建模相关改进
    """
    def __init__(
        self, 
        src_input_dim, 
        tgt_input_dim,
        hidden_dims=[128, 64, 32],  # 增加模型深度
        output_dim=64,  # 增加特征维度
        use_batch_norm=True,
        dropout=0.1,  # 调整dropout率
        use_attention=True,  # 启用注意力机制
        use_residual=True,  # 启用残差连接
        use_feature_interaction=True,  # 启用特征交互
        temperature=0.07,  # 对比学习温度参数
        use_contrastive_loss=True  # 是否使用对比学习损失
    ):
        super().__init__()
        # 源域和目标域特征已经对齐，使用相同的维度
        input_dim = src_input_dim
        
        # 1. 增强的特征编码器（结合自注意力和残差连接）
        self.encoder = FeatureEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            use_attention=use_attention,
            use_residual=use_residual,
            use_feature_interaction=use_feature_interaction
        )
        
        # 2. 金融风险分类器（添加风险校准层）
        self.classifier = FraudClassifier(
            input_dim=output_dim,
            hidden_dims=[64, 32, 16],
            dropout=dropout * 0.5
        )
        
        # 3. 域判别器（增强版）
        self.domain = DomainDiscriminator(
            input_dim=output_dim,
            hidden_dims=[128, 64, 32, 16],
            dropout=dropout
        )
        
        # 4. 对比学习相关参数
        self.use_contrastive_loss = use_contrastive_loss
        self.temperature = temperature
        
        # 5. 金融特征工程模块
        self.financial_feature_processor = FinancialFeatureProcessor(
            input_dim=output_dim,
            hidden_dim=output_dim // 2
        )
        
        # 6. 风险校准层
        self.risk_calibrator = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, lambda_=1.0):
        # 特征编码
        features = self.encoder(x)
        
        # 金融特征处理
        features = self.financial_feature_processor(features)
        
        # 分类预测
        class_pred = self.classifier(features)
        
        # 风险校准
        calibrated_pred = self.risk_calibrator(class_pred)
        
        # 域判别预测
        domain_pred = self.domain(grl(features, lambda_))
        
        return calibrated_pred, domain_pred, features
    
    def contrastive_loss(self, src_features, tgt_features):
        """
        对比学习损失计算
        目标：拉近相同类别特征，推远不同类别特征
        """
        if not self.use_contrastive_loss:
            return 0.0
        
        # 合并源域和目标域特征
        all_features = torch.cat([src_features, tgt_features], dim=0)
        
        # 计算特征相似度矩阵
        similarity_matrix = torch.matmul(all_features, all_features.T) / self.temperature
        
        # 掩码：排除对角线元素（自身比较）
        mask = torch.eye(all_features.size(0), dtype=torch.bool).to(all_features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # 计算对比损失
        labels = torch.cat([torch.arange(src_features.size(0)), torch.arange(tgt_features.size(0))]).to(all_features.device)
        contrastive_loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
        
        return contrastive_loss