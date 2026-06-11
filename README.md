# MLP Quantization and Pruning Sensitivity Lab

![Quantization overview](assets/quantization.png)

Figure: quantization stores the same model weights with fewer bits, reducing memory while trying to keep the model useful.

![Quantization and pruning overview](assets/quantization-and-pruning.jpg)

Figure: quantization reduces numeric precision, while pruning removes small or less important weights.

![Project overview](assets/readme_project_overview.png)

Figure: train an MLP on real CIFAR-10, then test quantization, pruning, and combined pruning plus int8 compression.

## Motivation

A compression lab should show where accuracy stays stable and where it starts to break. This project uses real CIFAR-10, which is hard enough to make the compression curves meaningful.

## Project Goal

We trained a small MLP on real CIFAR-10 and measured how accuracy changes under:

- Weight quantization from 64-bit storage down to 1-bit
- Magnitude pruning from 0% to 98% sparsity
- Combined pruning followed by int8 quantization

## Dataset

We used the official CIFAR-10 Python archive. The script downloads it into the ignored `data/` folder.

Experiment subset:

- Training images: 5,000
- Test images: 1,500
- Classes: 10
- Image size: 32x32 RGB
- Features: flattened standardized pixels

## Tools

Python, NumPy, pandas, scikit-learn, and matplotlib.

## Model

The model is an MLP classifier with:

- Hidden layers: 128 and 64 neurons
- Activation: ReLU
- Optimizer: Adam
- Early stopping: enabled
- Validation fraction: 0.15
- Maximum iterations: 80

This is still not a CNN, so the baseline accuracy is modest. That is acceptable here because the goal is compression sensitivity, not state-of-the-art CIFAR-10 classification.

## Baseline Result

| Model | Accuracy | Macro F1 | Parameters |
|---|---:|---:|---:|
| Real CIFAR-10 MLP | 0.4080 | 0.4029 | 402,250 |

## Quantization Results

| Bits Per Parameter | Accuracy | Macro F1 | Compression Ratio |
|---:|---:|---:|---:|
| 64 | 0.4080 | 0.4029 | 1.00 |
| 32 | 0.4080 | 0.4029 | 2.00 |
| 16 | 0.4080 | 0.4029 | 4.00 |
| 8 | 0.4080 | 0.4032 | 8.00 |
| 4 | 0.3980 | 0.3951 | 16.00 |
| 2 | 0.1933 | 0.1592 | 32.00 |
| 1 | 0.2253 | 0.2009 | 64.00 |

![Accuracy vs quantization](results/accuracy_vs_quantization_bits.png)

## Pruning Results

| Weight Sparsity | Accuracy | Macro F1 | Compression Ratio |
|---:|---:|---:|---:|
| 0.00 | 0.4080 | 0.4029 | 0.98 |
| 0.25 | 0.4140 | 0.4094 | 1.31 |
| 0.50 | 0.4060 | 0.4044 | 1.94 |
| 0.70 | 0.4040 | 0.4005 | 3.16 |
| 0.85 | 0.3827 | 0.3752 | 6.02 |
| 0.90 | 0.3600 | 0.3521 | 8.62 |
| 0.95 | 0.3180 | 0.3071 | 15.13 |
| 0.98 | 0.1993 | 0.1737 | 27.69 |

![Accuracy vs pruning](results/accuracy_vs_pruning_sparsity.png)

![Compression trade-off](results/compression_accuracy_tradeoff.png)

## Interpretation

The real CIFAR-10 experiment gives a useful pattern. Int8 quantization preserved accuracy almost perfectly. Four-bit quantization caused a small drop. Two-bit and one-bit quantization damaged the model strongly.

Pruning was surprisingly stable up to about 70% sparsity. This suggests many weights in the MLP are not essential for this subset. After 85% sparsity, accuracy started to fall more clearly, and 98% pruning damaged the model heavily.

The combined pruning plus int8 result is practical: pruning 70% of weights and then using int8 kept accuracy at 0.4040 while giving an estimated 18.66x compression ratio.

## Conclusion

This project shows real compression sensitivity on CIFAR-10. For this MLP, int8 quantization and moderate pruning are safe, while very aggressive pruning or 2-bit/1-bit quantization is risky. A stronger next step is to repeat the same analysis on a small CNN.

## How To Run

```bash
pip install -r requirements.txt
python 1_real_cifar_mlp_compression_sensitivity.py
```
