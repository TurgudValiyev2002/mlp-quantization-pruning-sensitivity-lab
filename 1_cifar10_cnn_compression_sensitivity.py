from __future__ import annotations

import copy
import pickle
import tarfile
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE = DATA / "cifar-10-python.tar.gz"
EXTRACTED = DATA / "cifar-10-batches-py"
SEED = 42


class SmallCIFARCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.features(x), 1))


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def download_cifar10() -> None:
    DATA.mkdir(exist_ok=True)
    if not ARCHIVE.exists():
        urllib.request.urlretrieve(URL, ARCHIVE)
    if not EXTRACTED.exists():
        with tarfile.open(ARCHIVE, "r:gz") as tar:
            tar.extractall(DATA, filter="data")


def batch_path(name: str) -> Path:
    flat = DATA / name
    if flat.is_file():
        return flat
    standard = EXTRACTED / name
    if standard.is_file():
        return standard
    raise FileNotFoundError(name)


def load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="latin1")
    x = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(batch["labels"], dtype=np.int64)
    return x, y


def load_subset(train_per_class: int = 600, test_per_class: int = 150) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    download_cifar10()
    xs, ys = [], []
    for batch_id in range(1, 6):
        x, y = load_batch(batch_path(f"data_batch_{batch_id}"))
        xs.append(x)
        ys.append(y)
    x_train_all = np.concatenate(xs)
    y_train_all = np.concatenate(ys)
    x_test_all, y_test_all = load_batch(batch_path("test_batch"))
    rng = np.random.default_rng(SEED)

    def balanced(x: np.ndarray, y: np.ndarray, per_class: int) -> tuple[np.ndarray, np.ndarray]:
        selected = []
        for label in range(10):
            selected.extend(rng.choice(np.where(y == label)[0], size=per_class, replace=False))
        selected = np.array(selected)
        rng.shuffle(selected)
        return x[selected], y[selected]

    return (*balanced(x_train_all, y_train_all, train_per_class), *balanced(x_test_all, y_test_all, test_per_class))


def make_dataset(images: np.ndarray, labels: np.ndarray) -> TensorDataset:
    x = images.astype(np.float32) / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
    x = ((x - mean) / std).transpose(0, 3, 1, 2)
    return TensorDataset(torch.tensor(x), torch.tensor(labels, dtype=torch.long))


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    labels = []
    preds = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            labels.extend(y_batch.numpy())
            preds.extend(torch.argmax(logits, dim=1).numpy())
    return accuracy_score(labels, preds), f1_score(labels, preds, average="macro")


def train_model(train_loader: DataLoader, val_loader: DataLoader) -> tuple[nn.Module, pd.DataFrame, int]:
    set_seed()
    model = SmallCIFARCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    best_state = None
    best_val_acc = -1.0
    best_epoch = 0
    history = []
    for epoch in range(1, 7):
        model.train()
        train_labels = []
        train_preds = []
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            train_labels.extend(y_batch.numpy())
            train_preds.extend(torch.argmax(logits.detach(), dim=1).numpy())
        val_acc, _ = evaluate(model, val_loader)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / len(train_loader.dataset),
                "train_accuracy": accuracy_score(train_labels, train_preds),
                "val_accuracy": val_acc,
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch


def quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 32:
        return tensor.clone()
    if bits == 1:
        scale = tensor.abs().mean()
        return tensor.sign() * (scale if scale > 0 else 1.0)
    levels = 2 ** (bits - 1) - 1
    max_abs = tensor.abs().max()
    if max_abs == 0:
        return tensor.clone()
    scale = max_abs / levels
    return torch.round(tensor / scale).clamp(-levels, levels) * scale


def quantized_model(model: nn.Module, bits: int) -> nn.Module:
    q_model = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in q_model.named_parameters():
            if "weight" in name or "bias" in name:
                param.copy_(quantize_tensor(param, bits))
    return q_model


def pruned_model(model: nn.Module, sparsity: float) -> nn.Module:
    p_model = copy.deepcopy(model)
    weights = []
    for name, param in p_model.named_parameters():
        if "weight" in name and param.ndim >= 2:
            weights.append(param.detach().abs().flatten())
    threshold = torch.quantile(torch.cat(weights), sparsity)
    with torch.no_grad():
        for name, param in p_model.named_parameters():
            if "weight" in name and param.ndim >= 2:
                param.mul_((param.abs() > threshold).float())
    return p_model


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def nonzero_weight_count(model: nn.Module) -> int:
    count = 0
    for name, param in model.named_parameters():
        if "weight" in name and param.ndim >= 2:
            count += int(torch.count_nonzero(param).item())
    return count


def plot_outputs(quant_df: pd.DataFrame, prune_df: pd.DataFrame, combo_df: pd.DataFrame, history: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(quant_df["bits_per_parameter"], quant_df["accuracy"], marker="o", linewidth=2)
    plt.gca().invert_xaxis()
    plt.ylim(0, max(0.65, quant_df["accuracy"].max() + 0.08))
    plt.xlabel("Bits per parameter")
    plt.ylabel("Accuracy")
    plt.title("CNN Accuracy vs Quantization Bits")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS / "accuracy_vs_quantization_bits.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(prune_df["weight_sparsity"], prune_df["accuracy"], marker="o", linewidth=2)
    plt.ylim(0, max(0.65, prune_df["accuracy"].max() + 0.08))
    plt.xlabel("Weight sparsity")
    plt.ylabel("Accuracy")
    plt.title("CNN Accuracy vs Magnitude Pruning")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS / "accuracy_vs_pruning_sparsity.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.scatter(quant_df["compression_ratio_vs_float32"], quant_df["accuracy"], label="quantization", s=70)
    plt.scatter(prune_df["compression_ratio_vs_dense_float32"], prune_df["accuracy"], label="pruning", s=70)
    plt.scatter(combo_df["compression_ratio_vs_dense_float32"], combo_df["accuracy"], label="prune + int8", s=70)
    plt.xlabel("Estimated compression ratio")
    plt.ylabel("Accuracy")
    plt.title("CNN Compression and Accuracy Trade-off")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS / "compression_accuracy_tradeoff.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    boxes = [
        ("Real CIFAR-10", 0.15),
        ("Train small CNN", 0.40),
        ("Quantize\nor prune", 0.64),
        ("Compare accuracy\nand compression", 0.88),
    ]
    for text, x in boxes:
        ax.text(x, 0.55, text, ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef6ff", edgecolor="#336699"))
    for start, end in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(end[1] - 0.12, 0.55), xytext=(start[1] + 0.12, 0.55), arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("CNN pruning and quantization sensitivity workflow", fontsize=15)
    fig.tight_layout()
    fig.savefig(ASSETS / "readme_project_overview.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    x_train, y_train, x_test, y_test = load_subset()
    rng = np.random.default_rng(SEED)
    train_idx = []
    val_idx = []
    for label in range(10):
        idx = np.where(y_train == label)[0]
        rng.shuffle(idx)
        split = int(0.85 * len(idx))
        train_idx.extend(idx[:split])
        val_idx.extend(idx[split:])
    train_loader = DataLoader(make_dataset(x_train[np.array(train_idx)], y_train[np.array(train_idx)]), batch_size=128, shuffle=True)
    val_loader = DataLoader(make_dataset(x_train[np.array(val_idx)], y_train[np.array(val_idx)]), batch_size=256, shuffle=False)
    test_loader = DataLoader(make_dataset(x_test, y_test), batch_size=256, shuffle=False)
    model, history, best_epoch = train_model(train_loader, val_loader)
    base_acc, base_f1 = evaluate(model, test_loader)
    params = parameter_count(model)
    dense_bytes = params * 4

    quant_rows = []
    for bits in [32, 16, 8, 4, 2, 1]:
        q_model = model if bits == 32 else quantized_model(model, bits)
        acc, macro_f1 = evaluate(q_model, test_loader)
        bytes_used = params * bits / 8
        quant_rows.append(
            {
                "bits_per_parameter": bits,
                "accuracy": round(acc, 4),
                "macro_f1": round(macro_f1, 4),
                "approx_model_bytes": round(bytes_used, 1),
                "compression_ratio_vs_float32": round(dense_bytes / bytes_used, 2),
            }
        )

    prune_rows = []
    for sparsity in [0.0, 0.25, 0.50, 0.70, 0.85, 0.90, 0.95]:
        p_model = model if sparsity == 0 else pruned_model(model, sparsity)
        acc, macro_f1 = evaluate(p_model, test_loader)
        nonzero = nonzero_weight_count(p_model)
        mask_bytes = nonzero * 4 + params / 8
        prune_rows.append(
            {
                "weight_sparsity": sparsity,
                "nonzero_weights": nonzero,
                "accuracy": round(acc, 4),
                "macro_f1": round(macro_f1, 4),
                "estimated_sparse_bytes": round(mask_bytes, 1),
                "compression_ratio_vs_dense_float32": round(dense_bytes / mask_bytes, 2),
            }
        )

    combo_rows = []
    for sparsity in [0.50, 0.70, 0.85, 0.90]:
        p_model = pruned_model(model, sparsity)
        q_model = quantized_model(p_model, 8)
        acc, macro_f1 = evaluate(q_model, test_loader)
        nonzero = nonzero_weight_count(q_model)
        sparse_int8_bytes = nonzero + params / 8
        combo_rows.append(
            {
                "method": f"prune_{int(sparsity * 100)}pct_then_int8",
                "weight_sparsity": sparsity,
                "accuracy": round(acc, 4),
                "macro_f1": round(macro_f1, 4),
                "estimated_sparse_int8_bytes": round(sparse_int8_bytes, 1),
                "compression_ratio_vs_dense_float32": round(dense_bytes / sparse_int8_bytes, 2),
            }
        )

    baseline = pd.DataFrame(
        [
            {
                "model": "small_cifar10_cnn_float32",
                "accuracy": round(base_acc, 4),
                "macro_f1": round(base_f1, 4),
                "parameters": params,
                "approx_model_bytes": dense_bytes,
                "train_images": len(x_train),
                "test_images": len(x_test),
                "best_epoch": best_epoch,
            }
        ]
    )
    quant_df = pd.DataFrame(quant_rows)
    prune_df = pd.DataFrame(prune_rows)
    combo_df = pd.DataFrame(combo_rows)
    baseline.to_csv(RESULTS / "baseline_summary.csv", index=False)
    quant_df.to_csv(RESULTS / "quantization_sensitivity.csv", index=False)
    prune_df.to_csv(RESULTS / "pruning_sensitivity.csv", index=False)
    combo_df.to_csv(RESULTS / "combined_pruning_int8.csv", index=False)
    history.to_csv(RESULTS / "cnn_training_history.csv", index=False)
    torch.save(model.state_dict(), RESULTS / "float32_small_cifar_cnn.pt")
    plot_outputs(quant_df, prune_df, combo_df, history)
    print(baseline.to_string(index=False))
    print(quant_df.to_string(index=False))
    print(prune_df.to_string(index=False))
    print(combo_df.to_string(index=False))


if __name__ == "__main__":
    main()
