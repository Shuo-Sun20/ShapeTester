import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs):
    return keras.ops.bartlett(inputs)

# Test with eager tensor (dynamic)
eager_input = tf.constant(1)
dynamic_output = call_func(eager_input)
print(f"Dynamic output shape: {dynamic_output.shape}")

# Test with Keras.Input placeholder (static)
try:
    static_input = keras.Input(shape=(), dtype='int32')
    static_output = call_func(static_input)
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Static output shape error: {e}")

# Additional test to show the defect more clearly
print(f"Dynamic output: {dynamic_output}")
print(f"Dynamic output type: {type(dynamic_output)}")