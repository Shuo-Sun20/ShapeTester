import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, newshape):
    return keras.ops.reshape(inputs, newshape)

# Test with eager tensor
eager_tensor = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8],
                           [9, 10, 11, 12, 13, 14, 15, 16],
                           [17, 18, 19, 20, 21, 22, 23, 24],
                           [25, 26, 27, 28, 29, 30, 31, 32],
                           [33, 34, 35, 36, 37, 38, 39, 40],
                           [41, 42, 43, 44, 45, 46, 47, 48]], dtype=tf.float32)

print("Eager tensor shape:", eager_tensor.shape)

# Call with eager tensor
dynamic_result = call_func(eager_tensor, 48)
print("Dynamic output shape:", dynamic_result.shape)

# Test with Keras Input placeholder
input_placeholder = keras.Input(shape=(8,), batch_size=6)
print("Input placeholder shape:", input_placeholder.shape)

# Call with placeholder
try:
    static_result = call_func(input_placeholder, 48)
    print("Static output shape:", static_result.shape)
except Exception as e:
    print("Static output shape error:", str(e))