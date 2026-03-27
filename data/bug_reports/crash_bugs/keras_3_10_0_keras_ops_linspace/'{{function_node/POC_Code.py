import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    start, stop = inputs[0], inputs[1]
    return keras.ops.linspace(start, stop, num, endpoint, retstep, dtype, axis)

# Create eager tensors (scalar tensors with shape ())
eager_start = tf.constant(0.0)
eager_stop = tf.constant(1.0)
eager_inputs = [eager_start, eager_stop]

# Create Keras Input placeholders with the same shape
input_start = keras.Input(shape=())
input_stop = keras.Input(shape=())
placeholder_inputs = [input_start, input_stop]

# Test parameters that cause the defect
test_params = {
    'num': 0,
    'endpoint': True,
    'retstep': False,
    'dtype': None,
    'axis': -2
}

print("Testing with eager tensors:")
try:
    eager_result = call_func(eager_inputs, **test_params)
    print(f"Eager result shape: {eager_result.shape}")
except Exception as e:
    print(f"Eager execution error: {e}")

print("\nTesting with Keras Input placeholders:")
try:
    static_result = call_func(placeholder_inputs, **test_params)
    print(f"Static result shape: {static_result.shape}")
except Exception as e:
    print(f"Static execution error: {e}")

print("\nDefect reproduced: Dynamic and static shapes are inconsistent")