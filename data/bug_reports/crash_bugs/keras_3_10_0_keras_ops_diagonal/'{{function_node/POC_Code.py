import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, offset=0, axis1=0, axis2=1):
    return keras.ops.diagonal(inputs, offset, axis1, axis2)

# Create eager tensor with shape (2, 3, 4)
eager_tensor = tf.constant([[[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12]],
                           [[13, 14, 15, 16],
                            [17, 18, 19, 20],
                            [21, 22, 23, 24]]], dtype=tf.float32)

# Create Keras Input placeholder with same shape
input_placeholder = keras.Input(shape=(2, 3, 4))

print("Eager tensor shape:", eager_tensor.shape)
print("Input placeholder shape:", input_placeholder.shape)

# Test with eager tensor - this should cause the error
print("\nTesting with eager tensor:")
try:
    eager_result = call_func(eager_tensor, offset=-2, axis1=0, axis2=0)
    print("Eager result shape:", eager_result.shape)
except Exception as e:
    print("Error with eager tensor:", str(e))

# Test with Keras Input placeholder - this should work and show static shape
print("\nTesting with Keras Input placeholder:")
try:
    static_result = call_func(input_placeholder, offset=-2, axis1=0, axis2=0)
    print("Static result shape:", static_result.shape)
except Exception as e:
    print("Error with placeholder:", str(e))