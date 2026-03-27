import numpy as np
import keras

def call_func(inputs, transform, interpolation="bilinear", fill_mode="constant", fill_value=0, data_format=None):
    return keras.ops.image.affine_transform(
        images=inputs,
        transform=transform,
        interpolation=interpolation,
        fill_mode=fill_mode,
        fill_value=fill_value,
        data_format=data_format
    )

# Create random input tensors
x = np.random.random((2, 64, 80, 3))  # Batch of 2 RGB images
transform = np.array(
    [
        [1.5, 0, -20, 0, 1.5, -16, 0, 0],  # zoom transform
        [1, 0, -20, 0, 1, -16, 0, 0],  # translation transform
    ]
)

example_output = call_func(x, transform)