import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, axis=None, keepdims=False, initial=None):
    return keras.ops.min(x=inputs, axis=axis, keepdims=keepdims, initial=initial)

# Test with eager tensor
eager_tensor = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=tf.float32)
print("Eager tensor shape:", eager_tensor.shape)

# Call with eager tensor
dynamic_result = call_func(inputs=eager_tensor, axis=[-1, -2], keepdims=False, initial=None)
print("Dynamic output shape:", dynamic_result.shape)

# Test with Keras Input placeholder
input_placeholder = keras.Input(shape=(3, 4))
print("Input placeholder shape:", input_placeholder.shape)

# Call with Keras Input placeholder
try:
    static_result = call_func(inputs=input_placeholder, axis=[-1, -2], keepdims=False, initial=None)
    print("Static output shape:", static_result.shape)
except Exception as e:
    print("Static output error:", str(e))