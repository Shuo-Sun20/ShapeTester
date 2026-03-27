import keras
import numpy as np

def call_func(inputs, k):
    targets, predictions = inputs
    return keras.ops.in_top_k(targets, predictions, k)

targets = keras.ops.convert_to_tensor(np.array([2, 5, 3], dtype=np.int32))
predictions = keras.ops.convert_to_tensor(
    np.array([[0.1, 0.4, 0.6, 0.9, 0.5],
              [0.1, 0.7, 0.9, 0.8, 0.3],
              [0.1, 0.6, 0.9, 0.9, 0.5]], dtype=np.float32)
)
example_output = call_func([targets, predictions], k=3)