import keras
import numpy as np
import tensorflow as tf
import numpy as np
import keras
import tensorflow as tf

def call_func(inputs, axis=-1):
    return keras.ops.log_softmax(inputs, axis=axis)

# Test input that causes the defect
test_input_shape = (2, 3)
test_axis = -3

# Create test data
test_data = np.random.randn(*test_input_shape)

# Test with eager tensor (dynamic)
print("Testing with eager tensor:")
try:
    eager_tensor = tf.convert_to_tensor(test_data)
    dynamic_result = call_func(eager_tensor, axis=test_axis)
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic output error: {e}")

# Test with Keras.Input placeholder (static)
print("\nTesting with Keras.Input placeholder:")
try:
    input_placeholder = keras.Input(shape=test_input_shape[1:])  # Exclude batch dimension
    static_result = call_func(input_placeholder, axis=test_axis)
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output error: {e}")

# Additional verification
print(f"\nTest input shape: {test_input_shape}")
print(f"Test axis: {test_axis}")