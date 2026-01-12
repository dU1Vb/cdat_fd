import torch
import torch.nn as nn
import torch.optim as optim

class FineTuneTrainer:
    def __init__(
        self,
        model,
        train_loader,
        device="cuda",
        lr=1e-4,
        freeze_encoder=False
    ):
        self.model = model.to(device)
        self.loader = train_loader
        self.device = device

        if freeze_encoder:
            for p in self.model.encoder.parameters():
                p.requires_grad = False

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )

    def train(self, epochs=10):
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for x, y in self.loader:
                x = x.to(self.device)
                y = y.to(self.device).float().view(-1, 1)

                pred = self.model(x)
                loss = self.criterion(pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"[Fine-tune][Epoch {epoch+1}] Loss: {total_loss:.4f}")
