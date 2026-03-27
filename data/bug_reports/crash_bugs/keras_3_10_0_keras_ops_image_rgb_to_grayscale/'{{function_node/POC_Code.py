import keras
import numpy as np
import tensorflow as tf
import numpy as np
import keras
from keras import ops

def call_func(inputs, data_format=None):
    return keras.ops.image.rgb_to_grayscale(images=inputs, data_format=data_format)

# Create test input - ndarray with shape (2, 4, 4, 3)
eager_input = np.random.random((2, 4, 4, 3)).astype(np.float32)

# Test with eager tensor
print("Testing with eager tensor:")
try:
    dynamic_output = call_func(eager_input, data_format='channels_first')
    print(f"Dynamic output shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic output error: {e}")

# Test with Keras.Input placeholder of same shape
print("\nTesting with Keras.Input placeholder:")
placeholder_input = keras.Input(shape=(4, 4, 3), batch_size=2)
static_output = call_func(placeholder_input, data_format='channels_first')
print(f"Static output shape: {static_output.shape}")

# Show the inconsistency
print(f"\nDefect reproduced:")
print(f"Input shape: (2, 4, 4, 3)")
print(f"data_format: 'channels_first'")
print(f"Expected behavior: Both should produce same shape")
print(f"Actual behavior: Different shapes/errors between eager and symbolic execution")