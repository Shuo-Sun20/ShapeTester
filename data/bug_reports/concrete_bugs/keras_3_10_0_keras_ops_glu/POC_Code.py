import keras
import numpy as np
import tensorflow as tf
import numpy as np
import keras

def call_func(inputs, axis=-1):
    if isinstance(inputs, list) and len(inputs) == 1:
        inputs = inputs[0]
    return keras.ops.glu(inputs, axis)

# Test with eager tensor
eager_input = np.random.randn(4, 8)
eager_output = call_func(eager_input, axis=-1)
dynamic_shape = list(eager_output.shape)
dynamic_shape[0] = None  # Replace batch dimension with None
print(f"Dynamic output shape: {dynamic_shape}")

# Test with Keras Input placeholder
placeholder_input = keras.Input(shape=(8,))
static_output = call_func(placeholder_input, axis=-1)
static_shape = list(static_output.shape)
static_shape[0] = None  # Replace batch dimension with None
print(f"Static output shape: {static_shape}")

# Verify the defect
print(f"Shapes match: {dynamic_shape == static_shape}")