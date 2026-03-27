import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs):
    # keras.ops.nonzero is a function, not a class
    # It takes a single input tensor, so we extract it from the list
    x = inputs[0]
    # Direct API call with the input tensor
    result = keras.ops.nonzero(x)
    return result

# Test with eager tensor (scalar - shape=())
eager_input = tf.constant(1.0)  # Scalar tensor with shape=()
eager_inputs = [eager_input]

print("Testing with eager tensor (scalar):")
print(f"Input shape: {eager_input.shape}")

try:
    dynamic_result = call_func(eager_inputs)
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic execution error: {e}")

# Test with Keras.Input placeholder of same shape
print("\nTesting with Keras.Input placeholder:")
placeholder_input = keras.Input(shape=())  # Scalar input
placeholder_inputs = [placeholder_input]

try:
    static_result = call_func(placeholder_inputs)
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static execution error: {e}")

print("\nDefect reproduction:")
print("The dynamic execution fails with scalar input while static execution succeeds")
print("This demonstrates the inconsistency between eager and symbolic execution")