import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, num=50, endpoint=True, base=10, dtype=None, axis=0):
    start, stop = inputs[0], inputs[1]
    return keras.ops.logspace(start, stop, num, endpoint, base, dtype, axis)

# Create eager tensors with shape (1,)
eager_start = tf.constant([1.0])
eager_stop = tf.constant([2.0])
eager_inputs = [eager_start, eager_stop]

# Create Keras.Input placeholders with the same shape
input_start = keras.Input(shape=(1,))
input_stop = keras.Input(shape=(1,))
placeholder_inputs = [input_start, input_stop]

# Test parameters that cause the defect
test_params = {
    'num': 0,
    'endpoint': True,
    'base': 10,
    'dtype': None,
    'axis': -2
}

# Call with eager tensors to get dynamic output shape
dynamic_output = call_func(eager_inputs, **test_params)
print(f"Dynamic output shape: {dynamic_output.shape}")

# Call with placeholders to get static output shape
static_output = call_func(placeholder_inputs, **test_params)
print(f"Static output shape: {static_output.shape}")

# Verify the shapes are different
print(f"Shapes are different: {dynamic_output.shape != static_output.shape}")