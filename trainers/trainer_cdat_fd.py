import torch
import torch.nn as nn
import torch.optim as optim
import math
from itertools import cycle

def lambda_schedule(epoch, max_epoch):
    """
    DANN 中经典的 lambda 调度策略
    """
    p = epoch / max_epoch
    return 2. / (1. + math.exp(-10 * p)) - 1

class CDATFDTrainer:
    def __init__(
        self,
        model,
        src_loader,
        tgt_loader,
        device="cuda",
        lr=1e-3
    ):
        self.model = model.to(device)
        self.src_loader = src_loader
        self.tgt_loader = tgt_loader
        self.device = device

        self.cls_criterion = nn.BCELoss()
        self.dom_criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs=50):
        self.model.train()

        tgt_iter = cycle(self.tgt_loader)

        for epoch in range(num_epochs):
            lambda_ = lambda_schedule(epoch, num_epochs)

            total_cls_loss = 0.0
            total_dom_loss = 0.0

            for xs, ys in self.src_loader:
                xt = next(tgt_iter)

                xs = xs.to(self.device)
                ys = ys.to(self.device).float().view(-1, 1)
                xt = xt.to(self.device)

                # ===== 源域 =====
                class_pred_s, domain_pred_s, src_features = self.model(xs, lambda_)
                cls_loss = self.cls_criterion(class_pred_s, ys)

                domain_label_s = torch.zeros(xs.size(0), dtype=torch.long).to(self.device)
                dom_loss_s = self.dom_criterion(domain_pred_s, domain_label_s)

                # ===== 目标域 =====
                _, domain_pred_t, tgt_features = self.model(xt, lambda_)
                domain_label_t = torch.ones(xt.size(0), dtype=torch.long).to(self.device)
                dom_loss_t = self.dom_criterion(domain_pred_t, domain_label_t)

                dom_loss = dom_loss_s + dom_loss_t
                
                # ===== 对比学习损失 =====
                contrastive_loss = 0.0
                if hasattr(self.model, 'contrastive_loss'):
                    contrastive_loss = self.model.contrastive_loss(src_features, tgt_features)
                
                # 调整损失权重
                loss = cls_loss * 100 + lambda_ * dom_loss * 0.1 + contrastive_loss * 0.5

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_cls_loss += cls_loss.item()
                total_dom_loss += dom_loss.item()

            print(
                f"[Epoch {epoch+1}/{num_epochs}] "
                f"Cls Loss: {total_cls_loss:.4f} | "
                f"Dom Loss: {total_dom_loss:.4f} | "
                f"Lambda: {lambda_:.4f}"
            )
