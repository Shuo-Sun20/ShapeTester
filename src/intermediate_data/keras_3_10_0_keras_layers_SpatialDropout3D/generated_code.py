import keras
import numpy as np

def call_func(
    rate,
    data_format=None,
    seed=None,
    name=None,
    dtype=None,
    inputs=None,
    training=None
):
    layer = keras.layers.SpatialDropout3D(
        rate=rate,
        data_format=data_format,
        seed=seed,
        name=name,
        dtype=dtype
    )
    return layer(inputs, training=training)

np.random.seed(42)
example_input = np.random.randn(2, 4, 8, 8, 8).astype(np.float32)
example_output = call_func(
    rate=0.5,
    inputs=example_input,
    training=True
)