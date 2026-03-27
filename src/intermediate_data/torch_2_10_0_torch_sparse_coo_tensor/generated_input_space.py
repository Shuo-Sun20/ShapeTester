import torch
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [torch.tensor([[0, 1, 2], [2, 0, 1]]), torch.randn(3)],
    "size": [3, 3],
    "dtype": None,
    "device": None,
    "pin_memory": False,
    "requires_grad": False,
    "check_invariants": None,
    "is_coalesced": None
}

# Task 2 & 3: Parameters affecting output tensor shape (except "inputs") are:
# - size: Can be None, list, tuple, or torch.Size
# - dtype: Affects the data type of values, not the shape directly, but included as it affects tensor construction
# - device: Does not affect shape
# - pin_memory: Does not affect shape
# - requires_grad: Does not affect shape
# - check_invariants: Does not affect shape
# - is_coalesced: Does not affect shape

# Task 4: Define InputSpace dataclass with discretized value ranges
@dataclass
class InputSpace:
    # size parameter value space
    size: List[Optional[List[int]]] = field(default_factory=lambda: [
        None,  # Infer size from indices
        [3, 3],  # Original test case
        [2, 4],  # Smaller than inferred
        [5, 5],  # Square matrix
        [1, 10],  # Single row
        [10, 1],  # Single column
        [2, 3, 4],  # 3D tensor
        [0, 0],  # Edge case: empty tensor
        [100, 100],  # Large tensor
    ])
    
    # dtype parameter value space (discrete)
    dtype: List[Optional[torch.dtype]] = field(default_factory=lambda: [
        None,  # Infer from values
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.complex64,
    ])
    
    # device parameter value space (discrete)
    device: List[Optional[torch.device]] = field(default_factory=lambda: [
        None,  # Use default
        torch.device('cpu'),
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        torch.device('mps') if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else torch.device('cpu'),
    ])
    
    # pin_memory parameter value space (discrete)
    pin_memory: List[bool] = field(default_factory=lambda: [False, True])
    
    # requires_grad parameter value space (discrete)
    requires_grad: List[bool] = field(default_factory=lambda: [False, True])
    
    # check_invariants parameter value space
    check_invariants: List[Optional[bool]] = field(default_factory=lambda: [None, False, True])
    
    # is_coalesced parameter value space
    is_coalesced: List[Optional[bool]] = field(default_factory=lambda: [None, False, True])