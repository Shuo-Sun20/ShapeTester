import keras
import numpy as np
import tensorflow as tf
import keras
import numpy as np

def call_func(inputs, axis=None):
    if isinstance(inputs, list):
        if len(inputs) == 1:
            x = inputs[0]
            weights = None
        elif len(inputs) == 2:
            x, weights = inputs
        else:
            raise ValueError("Inputs list must contain 1 or 2 tensors")
    else:
        x = inputs
        weights = None
    
    return keras.ops.average(x=x, axis=axis, weights=weights)

# Test with eager tensors (dynamic case)
print("Testing with eager tensors:")
x_eager = keras.ops.convert_to_tensor(np.random.random((3, 4)))
weights_eager = keras.ops.convert_to_tensor(np.random.random((4,)))
inputs_eager = [x_eager, weights_eager]

try:
    result_eager = call_func(inputs_eager, axis=None)
    print(f"Dynamic output shape: {result_eager.shape}")
except Exception as e:
    print(f"Dynamic execution error: {e}")

# Test with Keras.Input placeholders (static case)
print("\nTesting with Keras.Input placeholders:")
x_placeholder = keras.Input(shape=(4,))
weights_placeholder = keras.ops.convert_to_tensor(np.random.random((4,)))
inputs_placeholder = [x_placeholder, weights_placeholder]

try:
    result_placeholder = call_func(inputs_placeholder, axis=None)
    print(f"Static output shape: {result_placeholder.shape}")
except Exception as e:
    print(f"Static execution error: {e}")