import tensorflow as tf

def call_func(operators, is_non_singular=None, is_self_adjoint=None, 
              is_positive_definite=None, is_square=None, name=None, inputs=None):
    linear_operator = tf.linalg.LinearOperatorComposition(
        operators=operators,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    return linear_operator.matmul(inputs)

# Create two random linear operators
matrix1 = tf.random.normal(shape=[2, 3])
matrix2 = tf.random.normal(shape=[3, 4])
operator1 = tf.linalg.LinearOperatorFullMatrix(matrix1)
operator2 = tf.linalg.LinearOperatorFullMatrix(matrix2)

# Create random input tensor
input_tensor = tf.random.normal(shape=[4, 5])

# Call the function
example_output = call_func(operators=[operator1, operator2], inputs=input_tensor)