import tensorflow as tf
import keras

def call_func(inputs, scale, offset=0.0, name=None, dtype=None):
    layer = keras.layers.Rescaling(scale=scale, offset=offset, name=name, dtype=dtype)
    return layer(inputs)

random_tensor = tf.random.uniform(shape=(4, 224, 224, 3), minval=0, maxval=255, dtype=tf.float32)
example_output = call_func(inputs=random_tensor, scale=1./255, offset=0.0)