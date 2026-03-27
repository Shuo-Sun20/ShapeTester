import keras
import numpy as np

def call_func(factor, fill_mode, interpolation, seed, fill_value, data_format, inputs, training=None):
    layer = keras.layers.RandomRotation(
        factor=factor,
        fill_mode=fill_mode,
        interpolation=interpolation,
        seed=seed,
        fill_value=fill_value,
        data_format=data_format
    )
    if training is not None:
        return layer(inputs, training=training)
    else:
        return layer(inputs)

# Generate random input tensor
batch_size = 2
height = 32
width = 32
channels = 3
inputs = np.random.randn(batch_size, height, width, channels).astype(np.float32)

# Call the function with appropriate parameters
example_output = call_func(
    factor=0.2,
    fill_mode="reflect",
    interpolation="bilinear",
    seed=42,
    fill_value=0.0,
    data_format="channels_last",
    inputs=inputs,
    training=True
)