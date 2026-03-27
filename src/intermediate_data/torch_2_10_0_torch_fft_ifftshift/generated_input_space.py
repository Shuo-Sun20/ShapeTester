import torch
from dataclasses import dataclass

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': torch.randn(10, 10),
    'dim': None
}

# Tasks 2 & 3: Parameters affecting output shape and their value spaces
# The only parameter other than 'inputs' is 'dim'. It does not affect output shape,
# as torch.fft.ifftshift only rearranges elements without altering tensor dimensions.
# However, 'dim' affects which dimensions are rearranged. Its value space includes:
# - None (default, all dimensions)
# - Single integer (dimension index)
# - Tuple of integers (multiple dimensions)
# Since it's discrete, we list all possible types with representative examples.

@dataclass
class InputSpace:
    # 'dim' is included as it affects the rearrangement operation.
    # Value space covers all legal scenarios:
    dim: list = None
    
    def __post_init__(self):
        # Define the value space for 'dim'
        # Using a dummy tensor to get valid dimension indices
        dummy = torch.randn(5, 5, 5)  # 3D tensor for example
        ndim = dummy.ndim
        
        # Possible values for 'dim':
        # 1. None (default)
        # 2. Single integer: from -ndim to ndim-1 (typical and boundary values)
        # 3. Tuple of integers: combinations of valid dimension indices
        single_ints = [
            None,  # default
            0, 1, 2,  # positive typical
            -1, -2, -3,  # negative typical
            ndim - 1, -ndim,  # boundaries
            (ndim - 1) // 2  # middle dimension
        ]
        # Remove duplicates and ensure valid indices
        single_ints = list(dict.fromkeys([i for i in single_ints if i is None or -ndim <= i < ndim]))
        
        # Tuples: combinations of 2 and 3 dimensions
        tuple_examples = [
            (0, 1), (0, 2), (1, 2),  # 2D combinations
            (0, 1, 2), (-1, -2, -3),  # 3D combinations
            (0, -1), (-ndim, ndim - 1)  # mixed positive/negative
        ]
        # Filter to valid tuples (all indices within bounds)
        tuple_examples = [
            tup for tup in tuple_examples 
            if all(-ndim <= idx < ndim for idx in tup)
        ]
        
        self.dim = single_ints + tuple_examples