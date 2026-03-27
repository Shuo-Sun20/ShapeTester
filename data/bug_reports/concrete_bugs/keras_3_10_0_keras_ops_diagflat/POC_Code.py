import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, k=0):
    return keras.ops.diagflat(x=inputs, k=k)

# Test with eager tensor
eager_input = tf.constant([1, 2, 3, 4])
dynamic_output = call_func(eager_input, k=-5)
print(f"Dynamic output shape: {dynamic_output.shape}")

# Test with Keras Input placeholder
static_input = keras.Input(shape=(4,))
static_output = call_func(static_input, k=-5)
print(f"Static output shape: {static_output.shape}")

# Verify the defect
print(f"Dynamic shape: [None, {dynamic_output.shape[1]}]")
print(f"Static shape: [None, {static_output.shape[1]}]")
print(f"Shapes match: {dynamic_output.shape[1] == static_output.shape[1]}")