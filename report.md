# One-Page Report: CNN Quantization and Pruning Sensitivity

## Motivation

We compared quantization and pruning on the same CNN to understand which compression method is safer for a real CIFAR-10 image model.

## Dataset

The experiment used 6,000 CIFAR-10 training images and 1,500 test images across 10 classes.

## Method

We trained a small CNN with three convolution blocks. Then we evaluated post-training quantization, magnitude pruning, and pruning followed by int8 quantization.

## Results

The baseline CNN reached 0.4780 accuracy. Int8 quantization kept 0.4747 accuracy, while 2-bit quantization dropped to 0.1793. Pruning was stable up to 50% sparsity and reached 0.4907 accuracy. At 70% sparsity, accuracy fell to 0.4080. At 85% sparsity, it collapsed to 0.1607.

The best combined compression setting was 50% pruning followed by int8 quantization: 0.4940 accuracy with an estimated 6.44x compression ratio.

## Interpretation

Moderate compression is safe, but extreme compression destroys the learned representation. Int8 quantization is the safest method here. Pruning can help at 50%, but high sparsity is risky.

## Conclusion

The project gives a CNN-based compression sensitivity analysis. For this model, int8 and 50% pruning are practical; 2-bit quantization and 85%+ pruning are too aggressive.
