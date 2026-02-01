import numpy as np

class Neuron:
    """
    A single neuron performing the affine transformation: z = w * x + b
    followed by a non-linear activation.
    """
    def __init__(self, input_size):
        # We initialize weights randomly to break symmetry
        # This is 'architectural' thinking: initialization matters!
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()

    def sigmoid(self, z):
        """ The mathematical 'squashing' function """
        return 1 / (1 + np.exp(-z))

    def forward(self, inputs):
        """
        The Forward Pass.
        Math: a = \sigma( w . x + b )
        """
        # 1. The Linear Step (Dot Product)
        z = np.dot(self.weights, inputs) + self.bias
        
        # 2. The Non-Linear Step (Activation)
        activation = self.sigmoid(z)
        
        return activation

# --- Experiment ---
# Simulating an input vector (e.g., [pixel_1, pixel_2, pixel_3])
x = np.array([0.5, -0.2, 0.1])

n = Neuron(input_size=3)
output = n.forward(x)

print(f"Input: {x}")
print(f"Neuron Weights: {n.weights}")
print(f"Output Activation: {output:.4f}")
