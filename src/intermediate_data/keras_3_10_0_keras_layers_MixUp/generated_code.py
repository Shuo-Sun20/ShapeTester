import keras
import numpy as np

def call_func(alpha, data_format, seed, inputs):
    mix_up_layer = keras.layers.MixUp(alpha=alpha, data_format=data_format, seed=seed)
    output_dict = mix_up_layer(inputs)
    return [output_dict["images"], output_dict["labels"]]

batch_size = 8
images = np.random.rand(batch_size, 32, 32, 3).astype(np.float32)
labels = keras.ops.cast(
    keras.ops.one_hot(
        np.random.randint(0, 10, size=(batch_size,)).flatten(), 
        10
    ), 
    "float32"
)
inputs = {"images": images, "labels": labels}
example_output = call_func(alpha=0.2, data_format=None, seed=None, inputs=inputs)