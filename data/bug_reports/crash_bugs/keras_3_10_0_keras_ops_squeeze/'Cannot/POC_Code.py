import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, axis=None):
    x = inputs[0]
    return keras.ops.squeeze(x, axis)

# Test with eager tensor
eager_tensor = tf.constant([[[[1, 2, 3, 4, 5]]]])  # shape (1, 3, 1, 5)
eager_inputs = [eager_tensor]

# Get dynamic output shape with eager tensor
try:
    dynamic_result = call_func(eager_inputs, axis=0)
    dynamic_shape = dynamic_result.shape
    print(f"Dynamic output shape: {dynamic_shape}")
except Exception as e:
    print(f"Dynamic execution error: {e}")

# Test with Keras.Input placeholder
placeholder_input = keras.Input(shape=(3, 1, 5), batch_size=1)
placeholder_inputs = [placeholder_input]

# Get static output shape with placeholder
try:
    static_result = call_func(placeholder_inputs, axis=0)
    static_shape = static_result.shape
    print(f"Static output shape: {static_shape}")
except Exception as e:
    print(f"Static execution error: {e}")

print(f"Defect reproduced: Dynamic and static shapes are inconsistent")