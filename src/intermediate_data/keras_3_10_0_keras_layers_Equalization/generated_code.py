import keras
import tensorflow as tf

def call_func(inputs, value_range=(0, 255), bins=256, data_format=None):
    equalization_layer = keras.layers.Equalization(value_range=value_range, bins=bins, data_format=data_format)
    output = equalization_layer(inputs)
    return output

random_tensor = tf.random.uniform(shape=(1, 28, 28, 3), minval=0, maxval=256, dtype=tf.float32)
example_output = call_func(random_tensor)