import tensorflow as tf
import numpy as np
import tensorflow as tf

def call_func(inputs, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, is_square=True, name=None):
    diag = inputs[0]
    operator = tf.linalg.LinearOperatorDiag(
        diag=diag,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    matmul_input = inputs[1]
    return operator.matmul(matmul_input)

# Create test inputs
diag_tensor = tf.constant([[1.0, 2.0, 3.0, 4.0], 
                          [2.0, 3.0, 4.0, 5.0], 
                          [3.0, 4.0, 5.0, 6.0]], dtype=tf.float32)  # shape (3, 4)
matmul_tensor = tf.constant([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                            [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]],
                            [[3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]], dtype=tf.float32)  # shape (3, 4, 2)

inputs = [diag_tensor, matmul_tensor]

# Test parameters that cause the defect
test_params = {
    'is_non_singular': False,
    'is_self_adjoint': True, 
    'is_positive_definite': False,
    'is_square': None,
    'name': "test_operator"  # Provide a name to avoid the ValueError
}

# Direct call - dynamic output shape
print("=== Direct call (dynamic) ===")
direct_result = call_func(inputs, **test_params)
print(f"Dynamic output shape: {direct_result.shape}")
print(f"Dynamic result:\n{direct_result}")

# tf.function call - static output shape  
print("\n=== tf.function call (static) ===")
tf_func = tf.function(call_func)
static_result = tf_func(inputs, **test_params)
print(f"Static output shape: {static_result.shape}")
print(f"Static result:\n{static_result}")

# Compare shapes
print(f"\n=== Shape comparison ===")
print(f"Direct call shape: {direct_result.shape}")
print(f"tf.function call shape: {static_result.shape}")
print(f"Shapes are equal: {direct_result.shape == static_result.shape}")