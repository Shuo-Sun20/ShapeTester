import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf
import numpy as np

def call_func(inputs, pad_width, mode="constant", constant_values=None):
    return keras.ops.pad(x=inputs, pad_width=pad_width, mode=mode, constant_values=constant_values)

# Create test input as eager tensor
eager_input = tf.constant(np.random.random((3, 4, 5)), dtype=tf.float64)

# Create Keras Input placeholder with same shape
placeholder_input = keras.Input(shape=(4, 5), batch_size=3)

# Test parameters
pad_width = 0
mode = "constant"
constant_values = 0.5

print("Testing with eager tensor:")
try:
    dynamic_result = call_func(eager_input, pad_width, mode, constant_values)
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic output shape: Exception encountered when calling Pad.call().")
    print(f"{e}")

print("\nTesting with Keras Input placeholder:")
try:
    static_result = call_func(placeholder_input, pad_width, mode, constant_values)
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output shape: Exception encountered")
    print(f"{e}")