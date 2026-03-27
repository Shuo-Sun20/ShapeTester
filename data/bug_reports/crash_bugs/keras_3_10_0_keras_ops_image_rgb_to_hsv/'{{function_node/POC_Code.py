import keras
import numpy as np
import tensorflow as tf
import numpy as np
import keras
from keras import ops

def call_func(inputs, data_format=None):
    return keras.ops.image.rgb_to_hsv(images=inputs, data_format=data_format)

# Create test input
test_input = np.random.random((4, 4, 3)).astype(np.float32)

# Test with eager tensor (dynamic)
print("Testing with eager tensor:")
try:
    dynamic_result = call_func(test_input, data_format="channels_first")
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic execution error: {e}")

# Test with Keras.Input placeholder (static)
print("\nTesting with Keras.Input placeholder:")
try:
    placeholder_input = keras.Input(shape=(4, 3), dtype='float32')
    static_result = call_func(placeholder_input, data_format="channels_first")
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static execution error: {e}")

print(f"\nInput shape: {test_input.shape}")
print(f"Data format: channels_first")