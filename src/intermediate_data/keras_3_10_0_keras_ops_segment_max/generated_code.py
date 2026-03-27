import keras
import numpy as np

def call_func(inputs, num_segments=None, sorted=False):
    data = inputs[0]
    segment_ids = inputs[1]
    return keras.ops.segment_max(data, segment_ids, num_segments, sorted)

# Generate random input tensors
np.random.seed(42)
data = keras.ops.convert_to_tensor(np.random.randn(6).astype(np.float32))
segment_ids = keras.ops.convert_to_tensor([0, 0, 1, 1, 2, 2])
example_output = call_func([data, segment_ids], num_segments=3)