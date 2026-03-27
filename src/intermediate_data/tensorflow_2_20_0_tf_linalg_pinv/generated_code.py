import tensorflow as tf
import numpy as np

def call_func(inputs, rcond=None, validate_args=False, name='pinv'):
    a = inputs[0]
    if rcond is not None:
        return tf.linalg.pinv(a, rcond=rcond, validate_args=validate_args, name=name)
    else:
        return tf.linalg.pinv(a, validate_args=validate_args, name=name)

input_tensor = tf.constant(np.random.randn(4, 3).astype(np.float32))
example_output = call_func([input_tensor])