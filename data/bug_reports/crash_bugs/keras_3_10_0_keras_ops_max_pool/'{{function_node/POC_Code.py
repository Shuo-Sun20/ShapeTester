import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf
import numpy as np

def call_func(inputs, pool_size, strides=None, padding="valid", data_format=None):
    output = keras.ops.max_pool(inputs, pool_size, strides, padding, data_format)
    return output

# Create eager tensor with shape (2, 5, 5, 3)
eager_tensor = tf.constant(np.random.random((2, 5, 5, 3)), dtype=tf.float32)

# Create Keras Input placeholder with same shape
input_placeholder = keras.Input(shape=(5, 5, 3))

# Test parameters
pool_size = 4
strides = 4
padding = 'valid'
data_format = 'channels_first'

print("Testing with eager tensor:")
print(f"Input shape: {eager_tensor.shape}")
try:
    dynamic_output = call_func(eager_tensor, pool_size, strides, padding, data_format)
    print(f"Dynamic output shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic output error: {e}")

print("\nTesting with Keras Input placeholder:")
print(f"Input shape: {input_placeholder.shape}")
try:
    static_output = call_func(input_placeholder, pool_size, strides, padding, data_format)
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Static output error: {e}")