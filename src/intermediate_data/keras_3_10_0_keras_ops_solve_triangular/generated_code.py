import keras
import numpy as np

def call_func(inputs, lower=False):
    a, b = inputs
    return keras.ops.solve_triangular(a, b, lower=lower)

np.random.seed(42)
M = 5
N = 3
# Generate random lower triangular matrix
a = np.random.randn(M, M)
a = np.tril(a)
# Generate random right-hand side
b = np.random.randn(M, N)
example_output = call_func([a, b], lower=True)