import keras
import tensorflow as tf

def call_func(rate, seed, inputs, training):
    gaussian_dropout_layer = keras.layers.GaussianDropout(rate=rate, seed=seed)
    return gaussian_dropout_layer(inputs, training=training)

random_tensor = tf.random.normal(shape=(2, 3))
example_output = call_func(rate=0.5, seed=None, inputs=random_tensor, training=True)