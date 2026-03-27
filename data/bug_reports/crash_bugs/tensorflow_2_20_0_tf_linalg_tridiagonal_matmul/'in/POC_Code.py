import tensorflow as tf
import numpy as np
import tensorflow as tf

def call_func(inputs, diagonals_format='compact', name=None):
    diagonals = inputs[0]
    rhs = inputs[1]
    return tf.linalg.tridiagonal_matmul(diagonals, rhs, diagonals_format=diagonals_format, name=name)

# Create test inputs based on the provided information
# inputs: [EagerTensor(shape=(3, 4)), EagerTensor(shape=(4, 3))]
# diagonals_format: 'sequence'
# name: 'test_op'

# For sequence format, diagonals should be a list of three tensors: [superdiag, maindiag, subdiag]
# The first tensor in inputs represents the diagonals in sequence format
superdiag = tf.constant([[1.0, 2.0, 3.0, 0.0]], dtype=tf.float32)  # shape (1, 4)
maindiag = tf.constant([[4.0, 5.0, 6.0, 7.0]], dtype=tf.float32)   # shape (1, 4)  
subdiag = tf.constant([[0.0, 8.0, 9.0, 10.0]], dtype=tf.float32)   # shape (1, 4)

# Reshape to match the expected input shape (3, 4)
diagonals_tensor = tf.stack([superdiag[0], maindiag[0], subdiag[0]], axis=0)  # shape (3, 4)

# Create rhs tensor with shape (4, 3)
rhs = tf.constant([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0], 
                   [7.0, 8.0, 9.0],
                   [10.0, 11.0, 12.0]], dtype=tf.float32)

inputs = [diagonals_tensor, rhs]

# Test direct call (dynamic execution)
print("Testing direct call:")
try:
    result_direct = call_func(inputs, diagonals_format='sequence', name='test_op')
    print(f"Dynamic output shape: {result_direct.shape}")
except Exception as e:
    print(f"Direct call error: {e}")

# Test tf.function call (static execution)
print("\nTesting tf.function call:")
try:
    compiled_func = tf.function(call_func)
    result_compiled = compiled_func(inputs, diagonals_format='sequence', name='test_op')
    print(f"Static output shape: {result_compiled.shape}")
except Exception as e:
    print(f"tf.function call error: {e}")