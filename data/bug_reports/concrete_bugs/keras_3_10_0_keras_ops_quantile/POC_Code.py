import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, q, axis=None, method="linear", keepdims=False):
    return keras.ops.quantile(x=inputs, q=q, axis=axis, method=method, keepdims=keepdims)

# Test with eager tensor
eager_tensor = tf.random.normal((5, 4, 3))
q = [0.0, 0.5, 1.0]
axis = [1, 2]
method = "midpoint"
keepdims = False

# Get dynamic output shape
dynamic_result = call_func(eager_tensor, q, axis, method, keepdims)
dynamic_shape = dynamic_result.shape
print(f"Dynamic output shape: {dynamic_shape}")

# Test with Keras.Input placeholder
input_placeholder = keras.Input(shape=(4, 3))
static_result = call_func(input_placeholder, q, axis, method, keepdims)
static_shape = static_result.shape
print(f"Static output shape: {static_shape}")

print(f"Shapes match: {dynamic_shape == static_shape}")