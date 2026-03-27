import tensorflow as tf
import numpy as np

def call_func(inputs, name=None):
    return tf.math.softsign(features=inputs, name=name)

example_output = call_func(inputs=tf.random.normal(shape=(3, 4)))