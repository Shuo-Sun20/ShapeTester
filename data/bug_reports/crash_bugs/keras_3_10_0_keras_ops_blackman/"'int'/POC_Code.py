import keras
import numpy as np
import tensorflow as tf
import keras

def call_func(inputs):
    return keras.ops.blackman(inputs)

# Test input that causes the defect
test_input = 1

# Test with eager tensor (dynamic)
eager_tensor = keras.ops.convert_to_tensor(test_input)
dynamic_output = call_func(eager_tensor)
print(f"Dynamic output shape: {dynamic_output.shape}")

# Test with Keras.Input placeholder (static)
try:
    static_input = keras.Input(shape=(), dtype='int32')
    static_output = call_func(static_input)
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Static output shape error: {e}")

# Demonstrate the inconsistency
print(f"Dynamic result: {dynamic_output}")
print(f"Dynamic shape: {dynamic_output.shape}")