import tensorflow as tf
import numpy as np
import tensorflow as tf
import numpy as np

def call_func(inputs, eigvals_only=True, select='a', select_range=None, tol=None, name=None):
    alpha = inputs[0]
    beta = inputs[1]
    result = tf.linalg.eigh_tridiagonal(
        alpha=alpha,
        beta=beta,
        eigvals_only=eigvals_only,
        select=select,
        select_range=select_range,
        tol=tol,
        name=name
    )
    return result

# Create test inputs
alpha = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
beta = tf.constant([0.5, 0.5], dtype=tf.float32)
inputs = [alpha, beta]

# Test parameters
eigvals_only = False
select = 'v'
select_range = [0, 1]
tol = None
name = None

print("Testing dynamic execution:")
try:
    dynamic_result = call_func(inputs, eigvals_only=eigvals_only, select=select, 
                              select_range=select_range, tol=tol, name=name)
    print(f"Dynamic output shapes: {[tensor.shape.as_list() for tensor in dynamic_result]}")
except Exception as e:
    print(f"Dynamic execution error: {e}")

print("\nTesting static execution with tf.function:")
try:
    static_func = tf.function(call_func)
    static_result = static_func(inputs, eigvals_only=eigvals_only, select=select,
                               select_range=select_range, tol=tol, name=name)
    print(f"Static output shapes: {[tensor.shape.as_list() for tensor in static_result]}")
except Exception as e:
    print(f"Static execution error: {e}")