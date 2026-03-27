import keras
import numpy as np

def call_func(inputs):
    # Split the combined inputs list into individual tensors
    input_tensor = inputs[0]
    indices = inputs[1]
    updates = inputs[2]
    
    # Call scatter_update with the split parameters
    return keras.ops.scatter_update(input_tensor, indices, updates)

# Create random inputs matching the first example from the documentation
input_tensor = np.zeros((4, 4, 4)).astype('float32')
indices = [[1, 2, 3], [0, 1, 3]]
updates = np.array([1., 1.], dtype='float32')

# Call the function and save the output
example_output = call_func([input_tensor, indices, updates])