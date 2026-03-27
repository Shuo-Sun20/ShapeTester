import tensorflow as tf
from dataclasses import dataclass, field
from typing import List

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

# 1. Define valid_test_case
matrix1 = tf.random.normal(shape=[2, 3])
matrix2 = tf.random.normal(shape=[3, 4])
operator1 = tf.linalg.LinearOperatorFullMatrix(matrix1)
operator2 = tf.linalg.LinearOperatorFullMatrix(matrix2)
input_tensor = tf.random.normal(shape=[4, 5])

valid_test_case = {
    "operators": [operator1, operator2],
    "is_non_singular": None,
    "is_self_adjoint": None,
    "is_positive_definite": None,
    "is_square": None,
    "name": None,
    "inputs": input_tensor
}

# 2. Only "operators" parameter affects output shape (excluding inputs)

# 3. Discretized value space for operators
def create_operator_list_examples():
    # Example 1: 2D case (2 operators)
    m1 = tf.random.normal(shape=[2, 3])
    m2 = tf.random.normal(shape=[3, 4])
    op1 = tf.linalg.LinearOperatorFullMatrix(m1)
    op2 = tf.linalg.LinearOperatorFullMatrix(m2)
    example1 = [op1, op2]
    
    # Example 2: Single operator
    m3 = tf.random.normal(shape=[5, 6])
    op3 = tf.linalg.LinearOperatorFullMatrix(m3)
    example2 = [op3]
    
    # Example 3: 3 operators chain
    m4 = tf.random.normal(shape=[2, 3])
    m5 = tf.random.normal(shape=[3, 4])
    m6 = tf.random.normal(shape=[4, 5])
    op4 = tf.linalg.LinearOperatorFullMatrix(m4)
    op5 = tf.linalg.LinearOperatorFullMatrix(m5)
    op6 = tf.linalg.LinearOperatorFullMatrix(m6)
    example3 = [op4, op5, op6]
    
    # Example 4: Batch operators (batch size 2)
    m7 = tf.random.normal(shape=[2, 2, 3])
    m8 = tf.random.normal(shape=[2, 3, 4])
    op7 = tf.linalg.LinearOperatorFullMatrix(m7)
    op8 = tf.linalg.LinearOperatorFullMatrix(m8)
    example4 = [op7, op8]
    
    # Example 5: Different batch sizes that broadcast (batch 1 and batch 3)
    m9 = tf.random.normal(shape=[1, 3, 4])
    m10 = tf.random.normal(shape=[3, 4, 5])
    op9 = tf.linalg.LinearOperatorFullMatrix(m9)
    op10 = tf.linalg.LinearOperatorFullMatrix(m10)
    example5 = [op9, op10]
    
    # Example 6: The valid_test_case example (must be included)
    example6 = [operator1, operator2]
    
    return [example1, example2, example3, example4, example5, example6]

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    operators: List[tf.linalg.LinearOperator] = field(
        default_factory=lambda: create_operator_list_examples()
    )