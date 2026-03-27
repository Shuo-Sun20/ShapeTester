import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, num=50, endpoint=True, base=10, dtype=None, axis=0):
    start, stop = inputs[0], inputs[1]
    return keras.ops.logspace(start, stop, num, endpoint, base, dtype, axis)

# Create eager tensors
eager_start = tf.constant([1.0], shape=(1,))
eager_stop = tf.constant([2.0], shape=(1,))
eager_inputs = [eager_start, eager_stop]

# Create Keras Input placeholders with same shape
static_start = keras.Input(shape=(1,))
static_stop = keras.Input(shape=(1,))
static_inputs = [static_start, static_stop]

# Test parameters that cause the defect
test_params = {
    'num': 0,
    'endpoint': True,
    'base': 10,
    'dtype': None,
    'axis': -3
}

print("Testing with eager tensors:")
try:
    dynamic_result = call_func(eager_inputs, **test_params)
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic execution error: {e}")

print("\nTesting with Keras Input placeholders:")
try:
    static_result = call_func(static_inputs, **test_params)
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static execution error: {e}")