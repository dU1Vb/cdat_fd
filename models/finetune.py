import torch.nn as nn
from models.cdat_fd import FeatureEncoder, FraudClassifier

class FineTuneModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = FeatureEncoder(input_dim)
        self.classifier = FraudClassifier()

    def forward(self, x):
        feat = self.encoder(x)
        return self.classifier(feat)
