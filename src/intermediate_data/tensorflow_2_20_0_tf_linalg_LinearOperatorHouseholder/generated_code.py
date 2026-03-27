import tensorflow as tf
import numpy as np

def call_func(reflection_axis, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, is_square=None, name=None, inputs=None):
    operator = tf.linalg.LinearOperatorHouseholder(
        reflection_axis=reflection_axis,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    return operator.matmul(inputs)

reflection_axis = tf.random.normal(shape=[5])
inputs = tf.random.normal(shape=[5, 3])
example_output = call_func(reflection_axis=reflection_axis, inputs=inputs)