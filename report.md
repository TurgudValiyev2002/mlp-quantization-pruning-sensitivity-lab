# Report: MLP Quantization and Pruning Sensitivity

## Motivation

We compared quantization and pruning because both are common model-compression methods for edge AI. Quantization lowers the number of bits per parameter, while pruning removes low-magnitude weights.

## Dataset

We used the same controlled CIFAR-style dataset as the previous quantization project: 1200 RGB images, 16x16x3 pixels, and 3 visual classes. This is not the real CIFAR dataset; it is a reproducible local proxy.

## Model

The model was an MLP classifier with hidden layers `(64, 32)`, ReLU activation, Adam optimizer, early stopping, and a 25 percent test split.

## Hyperparameters

The model used `alpha=1e-4`, `learning_rate_init=1e-3`, `max_iter=250`, and `random_state=42`. Quantization was tested at 32, 16, 8, 4, 2, and 1 bit. Pruning was tested from 0 percent to 98 percent weight sparsity.

## Results

The baseline MLP achieved 1.0000 accuracy and 1.0000 macro F1. Quantization kept full accuracy down to 4-bit. At 2-bit, accuracy decreased to 0.9767. At 1-bit, accuracy dropped to 0.7200.

Pruning kept full accuracy up to 95 percent sparsity. At 98 percent sparsity, accuracy collapsed to 0.3333.

The strongest combined result was 90 percent pruning followed by INT8 quantization. It kept 1.0000 accuracy with an estimated 33.34x compression ratio.

## Interpretation

INT8 and 4-bit quantization were safe in this task because the MLP decision boundary was robust. Very low-bit quantization damaged the model because too much weight precision was lost. Pruning worked well until the network became too sparse; after 98 percent pruning, too many useful connections were removed.

## Which Method Is Better?

In general, INT8 quantization is the safest first compression method because it is simple and hardware-friendly. Pruning can be better for high compression, but only when sparse storage and sparse inference are supported. In this experiment, the combined pruning-plus-INT8 method gave the best compression without accuracy loss.

## Conclusion

Quantization and pruning are not enemies. They compress models in different ways and can be combined. The best choice depends on hardware support, accuracy tolerance, and memory constraints.
