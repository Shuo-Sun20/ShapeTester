import keras
import keras.ops as kops

def call_func(inputs, start_indices, updates):
    return kops.slice_update(inputs, start_indices, updates)

# Generate random tensors
inputs = keras.random.uniform(shape=(5, 5))
start_indices = [3, 3]
updates = keras.random.uniform(shape=(2, 2))

# Call function and store result
example_output = call_func(inputs, start_indices, updates)