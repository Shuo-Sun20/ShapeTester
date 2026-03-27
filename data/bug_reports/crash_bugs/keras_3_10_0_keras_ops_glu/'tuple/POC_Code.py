import keras
import numpy as np
import tensorflow as tf
import keras
import numpy as np

def call_func(inputs, axis=-1):
    if isinstance(inputs, list) and len(inputs) == 1:
        inputs = inputs[0]
    return keras.ops.glu(inputs, axis)

# Test input that causes the defect
test_input = np.random.randn(4, 8)
axis = -3

# Test with eager tensors (dynamic)
print("Testing with eager tensors:")
try:
    dynamic_result = call_func(test_input, axis=axis)
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic output shape: {e}")

# Test with Keras.Input placeholders (static)
print("\nTesting with Keras.Input placeholders:")
try:
    placeholder_input = keras.Input(shape=(8,))
    static_result = call_func(placeholder_input, axis=axis)
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output shape: {e}")