import tensorflow as tf
import numpy as np
import tensorflow as tf

def call_func(
    inputs,
    is_non_singular=None,
    is_self_adjoint=None,
    is_positive_definite=None,
    is_square=None,
    name=None
):
    col, row = inputs[0], inputs[1]
    operator = tf.linalg.LinearOperatorToeplitz(
        col=col,
        row=row,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name or "LinearOperatorToeplitz"
    )
    x = inputs[2]
    return operator.matmul(x)

# Create test inputs
col = tf.constant([1., 2., 3.])
row = tf.constant([1., 4., -9.])
x = tf.constant([[1., 2.], [3., 4.], [5., 6.]])
inputs = [col, row, x]

# Test direct call
direct_result = call_func(
    inputs,
    is_non_singular=None,
    is_self_adjoint=False,
    is_positive_definite=False,
    is_square=None,
    name=None
)

# Test tf.function call
tf_function_call = tf.function(call_func)
function_result = tf_function_call(
    inputs,
    is_non_singular=None,
    is_self_adjoint=False,
    is_positive_definite=False,
    is_square=None,
    name=None
)

print("Direct call output shape:", direct_result.shape)
print("tf.function call output shape:", function_result.shape)
print("Shapes match:", direct_result.shape == function_result.shape)