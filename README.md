# neural-architecture-foundations
Studying the mathematical building blocks of neural networks from scratch

## 1. The Perceptron (The Atomic Unit)
The fundamental building block of this architecture is the single neuron. Mathematically, it performs an affine transformation followed by a non-linearity.

Given an input vector $x \in \mathbb{R}^n$, weight vector $w \in \mathbb{R}^n$, and bias $b \in \mathbb{R}$:

$$z = w \cdot x + b = \sum_{i=1}^{n} w_i x_i + b$$

The output activation $a$ is produced by the Sigmoid function $\sigma(z)$:

$$a = \sigma(z) = \frac{1}{1 + e^{-z}}$$
