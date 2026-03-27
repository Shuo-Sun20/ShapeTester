import keras
import keras.ops as ops

def call_func(index, branches, inputs):
    return ops.switch(index, branches, *inputs)

# Generate random tensors
x = keras.random.uniform(shape=(2, 3))
y = keras.random.uniform(shape=(2, 3))

# Define branch functions
add_fn = lambda a, b: a + b
subtract_fn = lambda a, b: a - b

# Create inputs list
inputs = [x, y]
branches = [add_fn, subtract_fn]

# Call function
example_output = call_func(0, branches, inputs)