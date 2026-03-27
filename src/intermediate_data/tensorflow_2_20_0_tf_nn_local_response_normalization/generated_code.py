import tensorflow as tf

def call_func(inputs, depth_radius=5, bias=1.0, alpha=1.0, beta=0.5, name=None):
    output = tf.nn.local_response_normalization(
        input=inputs,
        depth_radius=depth_radius,
        bias=bias,
        alpha=alpha,
        beta=beta,
        name=name
    )
    return output

# Generate random 4D tensor as input (batch, height, width, channels)
input_tensor = tf.random.normal(shape=(2, 4, 4, 3), dtype=tf.float32)

# Call the function and save output
example_output = call_func(inputs=input_tensor)