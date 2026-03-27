import keras
import tensorflow as tf
from dataclasses import dataclass, field

def call_func(rate, seed, inputs, training):
    gaussian_dropout_layer = keras.layers.GaussianDropout(rate=rate, seed=seed)
    return gaussian_dropout_layer(inputs, training=training)

# 1. Define valid_test_case
valid_test_case = {
    'rate': 0.5,
    'seed': None,
    'inputs': tf.random.normal(shape=(2, 3)),
    'training': True
}

# 2. Parameters that can affect output shape (except "inputs"): NONE
# GaussianDropout only applies multiplicative noise, so output shape is always same as input shape

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # Since no parameters affect output shape (except inputs), 
    # the class has no fields that affect shape
    pass