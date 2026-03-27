import torch
from dataclasses import dataclass
from typing import Union, Tuple, Optional

# 1. Valid test case
valid_test_case = {
    'output_size': (4, 8, 12),
    'inputs': torch.randn(2, 3, 8, 16, 32)
}

# 2. Parameter affecting output shape: output_size
# 3. Value space analysis:
#    - Type: Union[int, Tuple[Optional[int], Optional[int], Optional[int]]]
#    - Discrete values for cube dimensions: 1, 2, 4, 7, 10 (from example)
#    - Tuple combinations with None: 
#      * All ints: (1,2,3), (4,5,6), (7,8,9), (10,11,12), (1,1,1)
#      * Mixed None: (7,None,None), (None,5,None), (None,None,9), 
#        (4,None,6), (None,8,None), (None,None,None)
#    - Single int values: 1, 2, 4, 7, 10

@dataclass
class InputSpace:
    """Dataclass containing all parameters affecting output tensor shape."""
    
    output_size: list = None
    
    def __post_init__(self):
        if self.output_size is None:
            # Value space covering all legal scenarios
            self.output_size = [
                # Single int values (cube)
                1, 2, 4, 7, 10,
                
                # Tuple with all ints (from boundary to typical)
                (1, 1, 1),        # Minimum valid
                (1, 2, 3),        # Small asymmetric
                (4, 5, 6),        # Medium asymmetric
                (7, 8, 9),        # Medium-large asymmetric
                (10, 11, 12),     # Large asymmetric
                (4, 8, 12),       # Original test case
                
                # Tuple with None values (keep dimension unchanged)
                (7, None, None),  # From documentation example
                (None, 5, None),
                (None, None, 9),
                (4, None, 6),
                (None, 8, None),
                (None, None, None)  # Keep all dimensions
            ]