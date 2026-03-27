import keras
import numpy as np

def call_func(inputs, num_bins, mask_value=None, salt=None, output_mode='int', sparse=False):
    layer = keras.layers.Hashing(num_bins=num_bins, mask_value=mask_value, salt=salt, output_mode=output_mode, sparse=sparse)
    return layer(inputs)

# Generate random input data (list of random strings)
np.random.seed(42)
random_strings = [''.join(chr(np.random.randint(65, 91)) for _ in range(3)) for _ in range(5)]
example_input = [[s] for s in random_strings]
example_output = call_func(example_input, num_bins=10, output_mode='int')