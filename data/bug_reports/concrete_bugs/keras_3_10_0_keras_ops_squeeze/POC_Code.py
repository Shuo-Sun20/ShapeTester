import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, axis=None):
    x = inputs[0]
    return keras.ops.squeeze(x, axis)

# Test with eager tensor (dynamic)
eager_tensor = tf.constant([[[[1, 2, 3, 4, 5]]], [[[6, 7, 8, 9, 10]]], [[[11, 12, 13, 14, 15]]]])  # shape (3, 1, 1, 5)
eager_tensor = tf.expand_dims(eager_tensor, 0)  # shape (1, 3, 1, 5)
dynamic_result = call_func([eager_tensor], axis=None)
print(f"Dynamic output shape: {dynamic_result.shape}")

# Test with Keras Input placeholder (static)
input_placeholder = keras.Input(shape=(3, 1, 5), batch_size=1)
static_result = call_func([input_placeholder], axis=None)
print(f"Static output shape: {static_result.shape}")

print(f"Dynamic shape normalized: [None, {dynamic_result.shape[1]}]")
print(f"Static shape normalized: [None, {static_result.shape[1]}, {static_result.shape[2]}]")