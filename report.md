# One-Page Report: MLP Quantization and Pruning Sensitivity

## Motivation

We use real CIFAR-10 so the compression curves show meaningful accuracy changes.

## Dataset

We used the official CIFAR-10 Python archive. The experiment used 5,000 training images and 1,500 test images from 10 classes. Images were flattened into standardized pixel features.

## Model and Method

We trained an MLP with hidden layers `(128, 64)`, ReLU activation, Adam optimization, early stopping, validation fraction 0.15, and maximum 80 iterations. Then we evaluated quantization, magnitude pruning, and pruning followed by int8 quantization.

## Results

The baseline MLP achieved 0.4080 accuracy and 0.4029 macro F1. Int8 quantization preserved accuracy at 0.4080. Four-bit quantization dropped slightly to 0.3980. Two-bit quantization dropped to 0.1933.

Pruning was stable up to 70% sparsity: accuracy was 0.4040 at 70%. At 90% sparsity, accuracy fell to 0.3600, and at 98% sparsity it fell to 0.1993. Pruning 70% of weights followed by int8 quantization kept 0.4040 accuracy with an estimated 18.66x compression ratio.

## Interpretation

The model has redundancy: many small weights can be removed without large accuracy loss. However, very aggressive compression destroys useful signal. Int8 is safe here; 2-bit and 1-bit are not.

## Conclusion

This project gives a realistic compression sensitivity analysis. Moderate pruning and int8 quantization are useful, while extreme pruning or very low-bit quantization should be used carefully.
