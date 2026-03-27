import keras
import numpy as np

def call_func(inputs, interpolation="bilinear", fill_value=0, data_format=None):
    images, start_points, end_points = inputs
    return keras.ops.image.perspective_transform(
        images=images,
        start_points=start_points,
        end_points=end_points,
        interpolation=interpolation,
        fill_value=fill_value,
        data_format=data_format
    )

x = np.random.random((2, 64, 80, 3)).astype("float32")
start_points = np.array(
    [
        [[0, 0], [0, 64], [80, 0], [80, 64]],
        [[0, 0], [0, 64], [80, 0], [80, 64]],
    ]
).astype("float32")
end_points = np.array(
    [
        [[3, 5], [7, 64], [76, -10], [84, 61]],
        [[8, 10], [10, 61], [65, 3], [88, 43]],
    ]
).astype("float32")

example_output = call_func([x, start_points, end_points])