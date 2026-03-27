import keras
import tensorflow as tf

def call_func(
    rate,
    inputs,
    training,
    data_format=None,
    seed=None,
    name=None,
    dtype=None
):
    layer = keras.layers.SpatialDropout2D(
        rate=rate,
        data_format=data_format,
        seed=seed,
        name=name,
        dtype=dtype
    )
    return layer(inputs, training=training)

inputs = tf.random.normal(shape=(2, 4, 4, 3))
example_output = call_func(
    rate=0.5,
    inputs=inputs,
    training=True,
    data_format='channels_last'
)