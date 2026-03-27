import tensorflow as tf
import numpy as np
import tensorflow as tf
import numpy as np

def call_func(
    inputs,
    num_sampled,
    num_classes,
    num_true=1,
    sampled_values=None,
    remove_accidental_hits=True,
    seed=None,
    name=None
):
    weights, biases, labels, network_inputs = inputs
    return tf.nn.sampled_softmax_loss(
        weights=weights,
        biases=biases,
        labels=labels,
        inputs=network_inputs,
        num_sampled=num_sampled,
        num_classes=num_classes,
        num_true=num_true,
        sampled_values=sampled_values,
        remove_accidental_hits=remove_accidental_hits,
        seed=seed,
        name=name
    )

# Create test inputs based on the provided test case
weights = tf.constant(np.random.randn(10000, 128), dtype=tf.float32)
biases = tf.constant(np.random.randn(10000), dtype=tf.float32)
labels = tf.constant(np.random.randint(0, 10000, size=(32, 2)), dtype=tf.int64)
network_inputs = tf.constant(np.random.randn(32, 128), dtype=tf.float32)

inputs = [weights, biases, labels, network_inputs]

# Test parameters
num_sampled = 10
num_classes = 20000
num_true = 2
sampled_values = None
remove_accidental_hits = False
seed = None
name = None

print("Testing dynamic vs static output shapes...")

# Direct call (dynamic execution)
try:
    dynamic_result = call_func(
        inputs=inputs,
        num_sampled=num_sampled,
        num_classes=num_classes,
        num_true=num_true,
        sampled_values=sampled_values,
        remove_accidental_hits=remove_accidental_hits,
        seed=seed,
        name=name
    )
    print(f"Dynamic execution result shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic execution error: {e}")

# tf.function call (static execution)
@tf.function
def static_call_func(
    inputs,
    num_sampled,
    num_classes,
    num_true=1,
    sampled_values=None,
    remove_accidental_hits=True,
    seed=None,
    name=None
):
    return call_func(
        inputs=inputs,
        num_sampled=num_sampled,
        num_classes=num_classes,
        num_true=num_true,
        sampled_values=sampled_values,
        remove_accidental_hits=remove_accidental_hits,
        seed=seed,
        name=name
    )

try:
    static_result = static_call_func(
        inputs=inputs,
        num_sampled=num_sampled,
        num_classes=num_classes,
        num_true=num_true,
        sampled_values=sampled_values,
        remove_accidental_hits=remove_accidental_hits,
        seed=seed,
        name=name
    )
    print(f"Static execution result shape: {static_result.shape}")
except Exception as e:
    print(f"Static execution error: {e}")

print("\nDefect reproduction complete.")