import tensorflow as tf
import numpy as np

def call_func(num_rows, num_columns=None, batch_shape=None, dtype=tf.float32,
             is_non_singular=False, is_self_adjoint=True, 
             is_positive_definite=False, is_square=None, name=None,
             inputs=None):
    operator = tf.linalg.LinearOperatorZeros(
        num_rows=num_rows,
        num_columns=num_columns,
        batch_shape=batch_shape,
        dtype=dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    return operator.matmul(inputs)

input_tensor = tf.random.normal(shape=[2, 4])
example_output = call_func(num_rows=2, inputs=input_tensor)