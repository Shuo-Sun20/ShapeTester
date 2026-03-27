import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, ord=None, axis=None, keepdims=False):
    x = inputs[0]
    return keras.ops.norm(x, ord=ord, axis=axis, keepdims=keepdims)

# Test with eager tensors
eager_tensor = tf.constant([[1.0, 2.0, 3.0, 4.0], 
                           [5.0, 6.0, 7.0, 8.0], 
                           [9.0, 10.0, 11.0, 12.0]], dtype=tf.float32)
print("Eager tensor shape:", eager_tensor.shape)

# Call with eager tensor
dynamic_result = call_func([eager_tensor], ord=None, axis=[-1, -2], keepdims=False)
print("Dynamic output shape:", dynamic_result.shape)
print("Dynamic result:", dynamic_result)

# Test with Keras.Input placeholders
input_placeholder = keras.Input(shape=(3, 4))
print("Input placeholder shape:", input_placeholder.shape)

# Call with placeholder
try:
    static_result = call_func([input_placeholder], ord=None, axis=[-1, -2], keepdims=False)
    print("Static output shape:", static_result.shape)
except Exception as e:
    print("Static output shape error:", str(e))