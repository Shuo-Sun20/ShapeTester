import keras
import numpy as np

def call_func(inputs, segment_ids, num_segments=None, sorted=False):
    return keras.ops.segment_sum(inputs, segment_ids, num_segments, sorted)

data = keras.ops.convert_to_tensor(np.random.randn(6).astype(np.float32))
segment_ids = keras.ops.convert_to_tensor([0, 0, 1, 1, 2, 2])
example_output = call_func(data, segment_ids, num_segments=3)