import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, source, destination):
    return keras.ops.moveaxis(x=inputs, source=source, destination=destination)

# Test with eager tensor
eager_tensor = tf.constant([[[1, 2, 3, 4, 5],
                            [6, 7, 8, 9, 10],
                            [11, 12, 13, 14, 15],
                            [16, 17, 18, 19, 20]],
                           [[21, 22, 23, 24, 25],
                            [26, 27, 28, 29, 30],
                            [31, 32, 33, 34, 35],
                            [36, 37, 38, 39, 40]],
                           [[41, 42, 43, 44, 45],
                            [46, 47, 48, 49, 50],
                            [51, 52, 53, 54, 55],
                            [56, 57, 58, 59, 60]]])  # shape=(3, 4, 5)

# Call with eager tensor
dynamic_result = call_func(eager_tensor, source=0, destination=1)
print(f"Dynamic output shape: {dynamic_result.shape}")

# Test with Keras Input placeholder
input_placeholder = keras.Input(shape=(4, 5))  # shape=(None, 4, 5) with batch dimension
static_result = call_func(input_placeholder, source=0, destination=1)
print(f"Static output shape: {static_result.shape}")

# Demonstrate the inconsistency
print(f"Dynamic shape: {list(dynamic_result.shape)}")
print(f"Static shape: {list(static_result.shape)}")