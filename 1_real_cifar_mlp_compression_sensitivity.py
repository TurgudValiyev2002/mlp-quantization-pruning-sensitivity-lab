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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE = DATA / "cifar-10-python.tar.gz"
EXTRACTED = DATA / "cifar-10-batches-py"
SEED = 42


def download_cifar10() -> None:
    DATA.mkdir(exist_ok=True)
    if not ARCHIVE.exists():
        print(f"Downloading CIFAR-10 from {URL}")
        urllib.request.urlretrieve(URL, ARCHIVE)
    if not EXTRACTED.exists():
        with tarfile.open(ARCHIVE, "r:gz") as tar:
            tar.extractall(DATA, filter="data")


def load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="latin1")
    x = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(batch["labels"], dtype=np.int64)
    return x, y


def load_cifar10_subset(train_per_class: int = 500, test_per_class: int = 150) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    download_cifar10()
    train_images = []
    train_labels = []
    for batch_id in range(1, 6):
        x, y = load_batch(EXTRACTED / f"data_batch_{batch_id}")
        train_images.append(x)
        train_labels.append(y)
    x_train_all = np.concatenate(train_images)
    y_train_all = np.concatenate(train_labels)
    x_test_all, y_test_all = load_batch(EXTRACTED / "test_batch")
    rng = np.random.default_rng(SEED)

    def balanced_subset(x: np.ndarray, y: np.ndarray, per_class: int) -> tuple[np.ndarray, np.ndarray]:
        indices = []
        for label in range(10):
            label_idx = np.where(y == label)[0]
            indices.extend(rng.choice(label_idx, size=per_class, replace=False))
        indices = np.array(indices)
        rng.shuffle(indices)
        return x[indices], y[indices]

    return (*balanced_subset(x_train_all, y_train_all, train_per_class), *balanced_subset(x_test_all, y_test_all, test_per_class))


def pixel_features(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32).reshape(len(images), -1) / 255.0


def get_mlp(pipeline: Pipeline) -> MLPClassifier:
    return pipeline.named_steps["mlp"]


def parameter_count(model: MLPClassifier) -> int:
    return int(sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_))


def weight_count(model: MLPClassifier) -> int:
    return int(sum(w.size for w in model.coefs_))


def evaluate_model(model: MLPClassifier, x_test_scaled: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
    pred = model.predict(x_test_scaled)
    return accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")


def quantize_array(weights: np.ndarray, bits: int) -> np.ndarray:
    if bits == 64:
        return weights.copy()
    if bits == 1:
        scale = np.mean(np.abs(weights)) or 1.0
        return np.sign(weights) * scale
    levels = 2 ** (bits - 1) - 1
    max_abs = np.max(np.abs(weights))
    scale = max_abs / levels if max_abs > 0 else 1.0
    q = np.round(weights / scale).clip(-levels, levels)
    return q * scale


def quantized_model(model: MLPClassifier, bits: int) -> MLPClassifier:
    q_model = copy.deepcopy(model)
    q_model.coefs_ = [quantize_array(w, bits) for w in model.coefs_]
    q_model.intercepts_ = [quantize_array(b, bits) for b in model.intercepts_]
    return q_model


def pruned_model(model: MLPClassifier, sparsity: float) -> MLPClassifier:
    p_model = copy.deepcopy(model)
    all_weights = np.concatenate([np.abs(w).ravel() for w in model.coefs_])
    threshold = np.quantile(all_weights, sparsity)
    p_model.coefs_ = [np.where(np.abs(w) <= threshold, 0.0, w) for w in model.coefs_]
    return p_model


def plot_line(df: pd.DataFrame, x: str, y: str, path: Path, title: str, xlabel: str, invert_x: bool = False) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(df[x], df[y], marker="o", linewidth=2)
    if invert_x:
        plt.gca().invert_xaxis()
    plt.ylim(0, 0.45)
    plt.xlabel(xlabel)
    plt.ylabel(y.replace("_", " ").title())
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    x_train_img, y_train, x_test_img, y_test = load_cifar10_subset()
    x_train = pixel_features(x_train_img)
    x_test = pixel_features(x_test_img)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=80,
                    early_stopping=True,
                    n_iter_no_change=8,
                    validation_fraction=0.15,
                    random_state=SEED,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    scaler = pipeline.named_steps["scaler"]
    model = get_mlp(pipeline)
    x_test_scaled = scaler.transform(x_test)

    base_acc, base_f1 = evaluate_model(model, x_test_scaled, y_test)
    total_params = parameter_count(model)
    total_weights = weight_count(model)
    dense_float64_bytes = total_params * 8

    quant_rows = []
    for bits in [64, 32, 16, 8, 4, 2, 1]:
        q_model = model if bits == 64 else quantized_model(model, bits)
        acc, f1 = evaluate_model(q_model, x_test_scaled, y_test)
        approx_bytes = total_params * bits / 8
        quant_rows.append(
            {
                "bits_per_parameter": bits,
                "accuracy": round(acc, 4),
                "macro_f1": round(f1, 4),
                "approx_model_bytes": round(approx_bytes, 1),
                "compression_ratio_vs_float64": round(dense_float64_bytes / approx_bytes, 2),
            }
        )

    prune_rows = []
    for sparsity in [0.0, 0.25, 0.50, 0.70, 0.85, 0.90, 0.95, 0.98]:
        p_model = model if sparsity == 0 else pruned_model(model, sparsity)
        acc, f1 = evaluate_model(p_model, x_test_scaled, y_test)
        nonzero = sum(np.count_nonzero(w) for w in p_model.coefs_)
        sparse_bytes = nonzero * 8 + total_weights / 8 + sum(b.size for b in model.intercepts_) * 8
        prune_rows.append(
            {
                "weight_sparsity": sparsity,
                "nonzero_weights": int(nonzero),
                "accuracy": round(acc, 4),
                "macro_f1": round(f1, 4),
                "estimated_sparse_bytes": round(sparse_bytes, 1),
                "compression_ratio_vs_dense_float64": round(dense_float64_bytes / sparse_bytes, 2),
            }
        )

    combo_rows = []
    for sparsity in [0.50, 0.70, 0.85, 0.90]:
        p_model = pruned_model(model, sparsity)
        q_model = quantized_model(p_model, 8)
        acc, f1 = evaluate_model(q_model, x_test_scaled, y_test)
        nonzero = sum(np.count_nonzero(w) for w in q_model.coefs_)
        sparse_int8_bytes = nonzero + total_weights / 8 + sum(b.size for b in model.intercepts_) * 8
        combo_rows.append(
            {
                "method": f"prune_{int(sparsity * 100)}pct_then_int8",
                "weight_sparsity": sparsity,
                "accuracy": round(acc, 4),
                "macro_f1": round(f1, 4),
                "estimated_sparse_int8_bytes": round(sparse_int8_bytes, 1),
                "compression_ratio_vs_dense_float64": round(dense_float64_bytes / sparse_int8_bytes, 2),
            }
        )

    summary = pd.DataFrame(
        [
            {
                "model": "real_cifar10_mlp_float64_storage",
                "accuracy": round(base_acc, 4),
                "macro_f1": round(base_f1, 4),
                "parameters": total_params,
                "weights": total_weights,
                "approx_model_bytes": dense_float64_bytes,
                "hidden_layers": "(128, 64)",
                "train_images": len(x_train),
                "test_images": len(x_test),
            }
        ]
    )
    quant_df = pd.DataFrame(quant_rows)
    prune_df = pd.DataFrame(prune_rows)
    combo_df = pd.DataFrame(combo_rows)
    summary.to_csv(RESULTS / "baseline_summary.csv", index=False)
    quant_df.to_csv(RESULTS / "quantization_sensitivity.csv", index=False)
    prune_df.to_csv(RESULTS / "pruning_sensitivity.csv", index=False)
    combo_df.to_csv(RESULTS / "combined_pruning_int8.csv", index=False)

    plot_line(
        quant_df,
        "bits_per_parameter",
        "accuracy",
        RESULTS / "accuracy_vs_quantization_bits.png",
        "Real CIFAR-10 Accuracy vs Quantization Bits",
        "Bits per parameter",
        invert_x=True,
    )
    plot_line(
        prune_df,
        "weight_sparsity",
        "accuracy",
        RESULTS / "accuracy_vs_pruning_sparsity.png",
        "Real CIFAR-10 Accuracy vs Weight Pruning",
        "Weight sparsity",
    )

    plt.figure(figsize=(7, 4))
    plt.scatter(quant_df["compression_ratio_vs_float64"], quant_df["accuracy"], label="quantization", s=80)
    plt.scatter(prune_df["compression_ratio_vs_dense_float64"], prune_df["accuracy"], label="pruning", s=80)
    plt.scatter(combo_df["compression_ratio_vs_dense_float64"], combo_df["accuracy"], label="prune + int8", s=80)
    plt.xlabel("Compression ratio vs dense float64")
    plt.ylabel("Accuracy")
    plt.ylim(0, 0.45)
    plt.title("Real CIFAR-10 Compression and Accuracy Trade-off")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS / "compression_accuracy_tradeoff.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    boxes = [
        ("Real CIFAR-10\nMLP baseline", 0.14),
        ("Quantization\n64 -> 1 bit", 0.38),
        ("Magnitude pruning\n0 -> 98%", 0.62),
        ("Compare accuracy\nand size", 0.86),
    ]
    for text, x in boxes:
        ax.text(x, 0.55, text, ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef6ff", edgecolor="#336699"))
    for start, end in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(end[1] - 0.11, 0.55), xytext=(start[1] + 0.11, 0.55), arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("MLP compression sensitivity workflow", fontsize=15)
    fig.tight_layout()
    fig.savefig(ASSETS / "readme_project_overview.png", dpi=180)
    plt.close(fig)

    print("Baseline")
    print(summary.to_string(index=False))
    print("\nQuantization sensitivity")
    print(quant_df.to_string(index=False))
    print("\nPruning sensitivity")
    print(prune_df.to_string(index=False))
    print("\nCombined pruning + int8")
    print(combo_df.to_string(index=False))


if __name__ == "__main__":
    main()
