import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, pool_size, strides=None, padding="valid", data_format=None):
    return keras.ops.average_pool(inputs, pool_size, strides, padding, data_format)

# Create eager tensor with shape (2, 8, 8, 3)
eager_tensor = tf.constant([[[[1.0, 2.0, 3.0] for _ in range(8)] for _ in range(8)] for _ in range(2)])

# Create Keras Input placeholder with same shape
input_placeholder = keras.Input(shape=(8, 8, 3))

# Test parameters
pool_size = 8
strides = 3
padding = "valid"
data_format = "channels_first"

print("Testing with eager tensor:")
try:
    dynamic_output = call_func(eager_tensor, pool_size, strides, padding, data_format)
    print(f"Dynamic output shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Error with eager tensor: {e}")

print("\nTesting with Keras Input placeholder:")
try:
    static_output = call_func(input_placeholder, pool_size, strides, padding, data_format)
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Error with placeholder: {e}")