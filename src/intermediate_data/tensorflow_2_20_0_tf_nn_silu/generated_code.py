import tensorflow as tf
import numpy as np

def call_func(inputs, beta=1.0):
    return tf.nn.silu(features=inputs, beta=beta)

example_output = call_func(tf.random.normal(shape=(3, 4)))