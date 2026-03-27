import tensorflow as tf

def call_func(inputs, compute_full_loss=False, name=None):
    targets, log_input = inputs
    return tf.nn.log_poisson_loss(
        targets=targets,
        log_input=log_input,
        compute_full_loss=compute_full_loss,
        name=name
    )

example_output = call_func(
    inputs=[tf.constant([1, 2, 3], dtype=tf.float32), 
            tf.constant([0.5, 1.2, -0.3], dtype=tf.float32)],
    compute_full_loss=True
)