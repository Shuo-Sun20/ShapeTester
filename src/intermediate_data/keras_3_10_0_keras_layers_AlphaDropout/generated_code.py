import keras
import numpy as np

def call_func(
    rate,
    noise_shape=None,
    seed=None,
    inputs=None,
    training=False
):
    layer = keras.layers.AlphaDropout(
        rate=rate,
        noise_shape=noise_shape,
        seed=seed
    )
    return layer(inputs, training=training)

# Generate random input tensor
np.random.seed(42)
input_tensor = np.random.randn(2, 10, 8).astype(np.float32)
example_output = call_func(
    rate=0.2,
    inputs=input_tensor,
    training=True
)