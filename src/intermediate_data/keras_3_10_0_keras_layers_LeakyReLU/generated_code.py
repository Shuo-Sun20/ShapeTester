import keras
import tensorflow as tf

def call_func(negative_slope, inputs, name=None, dtype=None):
    layer = keras.layers.LeakyReLU(negative_slope=negative_slope, name=name, dtype=dtype)
    return layer(inputs)

example_output = call_func(
    negative_slope=0.2,
    inputs=tf.random.uniform(shape=(2, 5), minval=-1.0, maxval=1.0),
    name="leaky_relu_example",
    dtype="float32"
)