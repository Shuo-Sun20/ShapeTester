import keras
import numpy as np

def call_func(num_bins, output_mode, sparse, name, dtype, inputs):
    layer = keras.layers.HashedCrossing(
        num_bins=num_bins,
        output_mode=output_mode,
        sparse=sparse,
        name=name,
        dtype=dtype
    )
    return layer(inputs)

# Generate random input tensors
np.random.seed(42)
feat1 = np.random.choice(['A', 'B', 'C'], size=5)
feat2 = np.random.choice([101, 102, 103], size=5)

example_output = call_func(
    num_bins=5,
    output_mode="int",
    sparse=False,
    name=None,
    dtype=None,
    inputs=[feat1, feat2]
)