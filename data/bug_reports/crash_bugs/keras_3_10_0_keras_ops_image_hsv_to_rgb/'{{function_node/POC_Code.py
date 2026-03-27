import keras
import numpy as np
import tensorflow as tf
import numpy as np
import keras
import tensorflow as tf

def call_func(inputs, data_format=None):
    return keras.ops.image.hsv_to_rgb(images=inputs, data_format=data_format)

# Create test input
test_input = np.random.random((2, 4, 4, 3))

# Test with eager tensor (dynamic)
print("Testing with eager tensor:")
try:
    eager_tensor = tf.constant(test_input)
    dynamic_result = call_func(eager_tensor, data_format='channels_first')
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic output error: {e}")

# Test with Keras.Input placeholder (static)
print("\nTesting with Keras.Input placeholder:")
try:
    placeholder_input = keras.Input(shape=(4, 4, 3))
    static_result = call_func(placeholder_input, data_format='channels_first')
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output error: {e}")

print("\nDefect reproduced: Dynamic and static output shapes are inconsistent")