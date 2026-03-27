import tensorflow as tf

def call_func(inputs, sample_weight=None, global_batch_size=None):
    # Split the inputs list into per_example_loss (the first tensor)
    per_example_loss = inputs[0]
    # Call the tf.nn.compute_average_loss API directly
    return tf.nn.compute_average_loss(
        per_example_loss,
        sample_weight=sample_weight,
        global_batch_size=global_batch_size
    )

# Generate random input tensors
per_example_loss = tf.random.normal(shape=[4, 1])
sample_weight = tf.random.uniform(shape=[4, 1])
# Create inputs list for call_func
inputs = [per_example_loss]
global_batch_size = 4

# Call the function and store the result
example_output = call_func(
    inputs=inputs,
    sample_weight=sample_weight,
    global_batch_size=global_batch_size
)