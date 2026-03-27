import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf
import numpy as np

def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.std(inputs, axis=axis, keepdims=keepdims)

# Create test input as eager tensor
eager_input = tf.constant(np.random.random((3, 4)), dtype=tf.float32)
print(f"Eager input shape: {eager_input.shape}")

# Test with eager tensor
dynamic_result = call_func(eager_input, axis=[-2, -1], keepdims=False)
print(f"Dynamic output shape: {dynamic_result.shape}")

# Test with Keras Input placeholder of same shape
static_input = keras.Input(shape=(3, 4))
try:
    static_result = call_func(static_input, axis=[-2, -1], keepdims=False)
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output shape error: {e}")

print(f"Dynamic result: {dynamic_result}")
print(f"Dynamic result shape: {list(dynamic_result.shape)}")