import torch
import torch.nn as nn
from models.cdat_fd import FeatureEncoder, FraudClassifier

class BaselineFraudModel(nn.Module):
    """
    改进的Baseline模型，使用深度特征编码器
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
        
        # 使用改进的特征编码器
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
        
        # 使用改进的分类器
        self.classifier = FraudClassifier(
            input_dim=output_dim,
            hidden_dims=[64, 32],
            dropout=dropout * 0.5
        )

    def forward(self, x):
        feat = self.encoder(x)
        return self.classifier(feat)
