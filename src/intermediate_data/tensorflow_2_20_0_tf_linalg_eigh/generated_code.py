import tensorflow as tf

def call_func(inputs, name=None):
    return tf.linalg.eigh(tensor=inputs, name=name)

# Generate random self-adjoint matrix
tf.random.set_seed(42)
batch_shape = (3, 4, 4)
rand_tensor = tf.random.normal(batch_shape)
# Make matrix self-adjoint: A = (M + M^T)/2
symmetric_tensor = (rand_tensor + tf.linalg.matrix_transpose(rand_tensor)) / 2.0

# Get eigenvalues and eigenvectors
eigenvalues, eigenvectors = call_func(inputs=symmetric_tensor)
# Store both outputs in a list as required
example_output = [eigenvalues, eigenvectors]