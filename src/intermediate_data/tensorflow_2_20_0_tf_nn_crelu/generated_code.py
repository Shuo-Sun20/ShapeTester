import tensorflow as tf
import numpy as np

def call_func(inputs, axis=-1, name=None):
    return tf.nn.crelu(features=inputs, axis=axis, name=name)

example_output = call_func(inputs=tf.random.normal(shape=(2, 5)))