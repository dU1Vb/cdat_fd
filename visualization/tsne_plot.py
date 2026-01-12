import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE

@torch.no_grad()
def extract_features(
    model,
    dataloader,
    device="cuda",
    max_samples=3000
):
    """
    提取 encoder 输出的潜在特征
    """
    model.eval()

    features = []
    labels = []
    domains = []

    count = 0
    for batch in dataloader:
        if len(batch) == 2:
            x, y = batch
        else:
            x = batch
            y = None

        x = x.to(device)

        # ⚠️ 只用 encoder
        z = model.encoder(x)

        features.append(z.cpu().numpy())

        if y is not None:
            labels.append(y.numpy())

        count += x.size(0)
        if count >= max_samples:
            break

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0) if labels else None

    return features, labels

def plot_tsne(
    features,
    domain_labels,
    title,
    save_path=None
):
    """
    domain_labels:
    0 -> Source Domain
    1 -> Target Domain
    """
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        random_state=42
    )

    emb = tsne.fit_transform(features)

    plt.figure(figsize=(7, 6))

    for domain, color, name in [
        (0, "tab:blue", "Source Domain"),
        (1, "tab:orange", "Target Domain")
    ]:
        idx = domain_labels == domain
        plt.scatter(
            emb[idx, 0],
            emb[idx, 1],
            s=10,
            alpha=0.6,
            c=color,
            label=name
        )

    plt.legend()
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

# 使用示例（需要在实际使用时取消注释并提供模型和数据加载器）:
# 
# # Source & Target 测试数据
# src_features, _ = extract_features(
#     baseline_model, source_test_loader, device="cuda"
# )
# tgt_features, _ = extract_features(
#     baseline_model, target_test_loader, device="cuda"
# )
# 
# features = np.vstack([src_features, tgt_features])
# 
# domain_labels = np.array(
#     [0] * len(src_features) + [1] * len(tgt_features)
# )
# 
# plot_tsne(
#     features,
#     domain_labels,
#     title="t-SNE Visualization (Before Alignment)",
#     save_path="tsne_before_alignment.png"
# )
# 
# src_features, _ = extract_features(
#     cdat_fd_model, source_test_loader, device="cuda"
# )
# tgt_features, _ = extract_features(
#     cdat_fd_model, target_test_loader, device="cuda"
# )
# 
# features = np.vstack([src_features, tgt_features])
# 
# domain_labels = np.array(
#     [0] * len(src_features) + [1] * len(tgt_features)
# )
# 
# plot_tsne(
#     features,
#     domain_labels,
#     title="t-SNE Visualization (After Alignment)",
#     save_path="tsne_after_alignment.png"
# )


def visualize_features(baseline_model, dann_model, source_loader, target_loader, device, save_dir):
    """可视化特征分布（t-SNE）"""
    print("\n" + "="*60)
    print("生成 t-SNE 可视化")
    print("="*60)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Baseline模型的特征
    print("[1] 提取Baseline模型特征...")
    src_features_baseline, _ = extract_features(baseline_model, source_loader, device, max_samples=2000)
    tgt_features_baseline, _ = extract_features(baseline_model, target_loader, device, max_samples=2000)
    
    features_baseline = np.vstack([src_features_baseline, tgt_features_baseline])
    domain_labels_baseline = np.array([0] * len(src_features_baseline) + [1] * len(tgt_features_baseline))
    
    plot_tsne(
        features_baseline,
        domain_labels_baseline,
        title="t-SNE Visualization (Baseline - Before Alignment)",
        save_path=str(save_dir / "tsne_baseline.png")
    )
    
    # DANN模型的特征
    print("[2] 提取DANN模型特征...")
    src_features_dann, _ = extract_features(dann_model, source_loader, device, max_samples=2000)
    tgt_features_dann, _ = extract_features(dann_model, target_loader, device, max_samples=2000)
    
    features_dann = np.vstack([src_features_dann, tgt_features_dann])
    domain_labels_dann = np.array([0] * len(src_features_dann) + [1] * len(tgt_features_dann))
    
    plot_tsne(
        features_dann,
        domain_labels_dann,
        title="t-SNE Visualization (DANN - After Alignment)",
        save_path=str(save_dir / "tsne_dann.png")
    )
    
    print(f"[OK] 可视化结果已保存到: {save_dir}")
