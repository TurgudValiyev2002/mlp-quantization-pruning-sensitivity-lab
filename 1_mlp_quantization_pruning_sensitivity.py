from __future__ import annotations

import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


RESULTS = Path("results")
RNG = np.random.default_rng(42)


def make_cifar_style_data(n: int = 1200, size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    for i in range(n):
        label = i % 3
        img = RNG.normal(0.25, 0.08, (size, size, 3))
        if label == 0:
            img[:, : size // 2, 0] += 0.45
        elif label == 1:
            img[: size // 2, :, 1] += 0.45
        else:
            rr, cc = np.ogrid[:size, :size]
            mask = (rr - size // 2) ** 2 + (cc - size // 2) ** 2 <= (size // 4) ** 2
            img[mask, 2] += 0.55
        images.append(np.clip(img, 0, 1))
        labels.append(label)
    return np.array(images), np.array(labels)


def parameter_count(model: MLPClassifier) -> int:
    return int(sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_))


def weight_count(model: MLPClassifier) -> int:
    return int(sum(w.size for w in model.coefs_))


def evaluate(model: MLPClassifier, x_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
    pred = model.predict(x_test)
    return accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")


def quantize_array(weights: np.ndarray, bits: int) -> np.ndarray:
    if bits == 1:
        scale = np.mean(np.abs(weights)) or 1.0
        return np.sign(weights) * scale
    levels = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(weights)) / levels if np.max(np.abs(weights)) > 0 else 1.0
    q = np.round(weights / scale).clip(-levels, levels)
    return q * scale


def quantized_model(model: MLPClassifier, bits: int) -> MLPClassifier:
    q_model = copy.deepcopy(model)
    q_model.coefs_ = [quantize_array(w, bits) for w in model.coefs_]
    return q_model


def pruned_model(model: MLPClassifier, sparsity: float) -> MLPClassifier:
    p_model = copy.deepcopy(model)
    all_weights = np.concatenate([np.abs(w).ravel() for w in model.coefs_])
    threshold = np.quantile(all_weights, sparsity)
    p_model.coefs_ = [np.where(np.abs(w) <= threshold, 0.0, w) for w in model.coefs_]
    return p_model


def plot_line(df: pd.DataFrame, x: str, y: str, path: Path, title: str, xlabel: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(df[x], df[y], marker="o", linewidth=2)
    plt.ylim(0, 1.05)
    plt.xlabel(xlabel)
    plt.ylabel(y.replace("_", " ").title())
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    images, y = make_cifar_style_data()
    x = images.reshape(len(images), -1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=250,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=15,
    )
    model.fit(x_train, y_train)

    base_acc, base_f1 = evaluate(model, x_test, y_test)
    total_params = parameter_count(model)
    total_weights = weight_count(model)
    baseline_bytes = total_params * 8

    quant_rows = []
    for bits in [32, 16, 8, 4, 2, 1]:
        q_model = model if bits == 32 else quantized_model(model, bits)
        acc, f1 = evaluate(q_model, x_test, y_test)
        approx_bytes = total_params * bits / 8
        quant_rows.append(
            {
                "bits_per_parameter": bits,
                "accuracy": round(acc, 4),
                "macro_f1": round(f1, 4),
                "approx_model_bytes": round(approx_bytes, 1),
                "compression_ratio_vs_float64": round(baseline_bytes / approx_bytes, 2),
            }
        )

    prune_rows = []
    for sparsity in [0.0, 0.25, 0.50, 0.70, 0.85, 0.90, 0.95, 0.98]:
        p_model = model if sparsity == 0 else pruned_model(model, sparsity)
        acc, f1 = evaluate(p_model, x_test, y_test)
        nonzero = sum(np.count_nonzero(w) for w in p_model.coefs_)
        sparse_bytes = nonzero * 8 + total_weights / 8 + sum(b.size for b in model.intercepts_) * 8
        prune_rows.append(
            {
                "weight_sparsity": sparsity,
                "nonzero_weights": int(nonzero),
                "accuracy": round(acc, 4),
                "macro_f1": round(f1, 4),
                "estimated_sparse_bytes": round(sparse_bytes, 1),
                "compression_ratio_vs_dense_float64": round(baseline_bytes / sparse_bytes, 2),
            }
        )

    combo_rows = []
    for sparsity in [0.5, 0.7, 0.85, 0.9]:
        p_model = pruned_model(model, sparsity)
        q_model = quantized_model(p_model, 8)
        acc, f1 = evaluate(q_model, x_test, y_test)
        nonzero = sum(np.count_nonzero(w) for w in q_model.coefs_)
        sparse_int8_bytes = nonzero + total_weights / 8 + sum(b.size for b in model.intercepts_) * 8
        combo_rows.append(
            {
                "method": f"prune_{int(sparsity * 100)}pct_then_int8",
                "weight_sparsity": sparsity,
                "accuracy": round(acc, 4),
                "macro_f1": round(f1, 4),
                "estimated_sparse_int8_bytes": round(sparse_int8_bytes, 1),
                "compression_ratio_vs_dense_float64": round(baseline_bytes / sparse_int8_bytes, 2),
            }
        )

    summary = pd.DataFrame(
        [
            {
                "model": "baseline_mlp_float64_storage",
                "accuracy": round(base_acc, 4),
                "macro_f1": round(base_f1, 4),
                "parameters": total_params,
                "weights": total_weights,
                "approx_model_bytes": baseline_bytes,
                "hidden_layers": "(64, 32)",
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
        quant_df.sort_values("bits_per_parameter", ascending=False),
        "bits_per_parameter",
        "accuracy",
        RESULTS / "accuracy_vs_quantization_bits.png",
        "Accuracy vs Quantization Bits",
        "Bits per parameter",
    )
    plot_line(
        prune_df,
        "weight_sparsity",
        "accuracy",
        RESULTS / "accuracy_vs_pruning_sparsity.png",
        "Accuracy vs Weight Pruning Sparsity",
        "Weight sparsity",
    )

    plt.figure(figsize=(7, 4))
    plt.scatter(quant_df["compression_ratio_vs_float64"], quant_df["accuracy"], label="quantization", s=80)
    plt.scatter(prune_df["compression_ratio_vs_dense_float64"], prune_df["accuracy"], label="pruning", s=80)
    plt.scatter(combo_df["compression_ratio_vs_dense_float64"], combo_df["accuracy"], label="prune + int8", s=80)
    plt.xlabel("Compression ratio vs dense float64")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.title("Accuracy and Compression Trade-off")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS / "compression_accuracy_tradeoff.png", dpi=160)
    plt.close()

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
