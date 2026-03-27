import tensorflow as tf

def call_func(
    inputs,
    input_output_dtype=None,
    is_non_singular=None,
    is_self_adjoint=None,
    is_positive_definite=None,
    is_square=True,
    name="LinearOperatorCirculant2D"
):
    spectrum = inputs[0]
    
    if input_output_dtype is None:
        input_output_dtype = spectrum.dtype
        
    operator = tf.linalg.LinearOperatorCirculant2D(
        spectrum=spectrum,
        input_output_dtype=input_output_dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    
    return operator.matvec(inputs[1])

spectrum = tf.complex(
    tf.random.normal([2, 3], dtype=tf.float32),
    tf.random.normal([2, 3], dtype=tf.float32)
)
vector = tf.complex(
    tf.random.normal([6], dtype=tf.float32),
    tf.random.normal([6], dtype=tf.float32)
)
example_output = call_func([spectrum, vector])