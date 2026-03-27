import keras
import numpy as np
import tensorflow as tf
import keras
import numpy as np

def call_func(inputs, from_logits=False, axis=-1):
    target, output = inputs
    return keras.ops.categorical_crossentropy(target, output, from_logits=from_logits, axis=axis)

# Test with eager tensors (dynamic)
print("Testing with eager tensors:")
target_eager = keras.ops.convert_to_tensor(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32))
output_eager = keras.ops.convert_to_tensor(np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]], dtype=np.float32))

print(f"Target eager shape: {target_eager.shape}")
print(f"Output eager shape: {output_eager.shape}")

try:
    result_eager = call_func([target_eager, output_eager], from_logits=False, axis=2)
    print(f"Dynamic output shape: {result_eager.shape}")
except Exception as e:
    print(f"Dynamic output error: {e}")

# Test with Keras.Input placeholders (static)
print("\nTesting with Keras.Input placeholders:")
target_input = keras.Input(shape=(3, 3))
output_input = keras.Input(shape=(3, 3))

print(f"Target input shape: {target_input.shape}")
print(f"Output input shape: {output_input.shape}")

try:
    result_static = call_func([target_input, output_input], from_logits=False, axis=2)
    print(f"Static output shape: {result_static.shape}")
except Exception as e:
    print(f"Static output error: {e}")