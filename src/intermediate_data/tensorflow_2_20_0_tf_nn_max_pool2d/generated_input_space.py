import tensorflow as tf
from dataclasses import dataclass, field
from typing import Union, List, Tuple

# 1. Define valid_test_case
valid_test_case = {
    'inputs': tf.random.normal(shape=(2, 8, 8, 3)),
    'ksize': (2, 2),
    'strides': (2, 2),
    'padding': 'VALID',
    'data_format': 'NHWC',
    'name': None
}

# 2. & 3. Identify shape-affecting parameters and their value spaces
@dataclass
class InputSpace:
    """
    Dataclass containing all parameters of tf.nn.max_pool2d (except 'inputs') 
    that affect output shape, with discretized value ranges.
    """
    
    # ksize: Window size for pooling. Affects receptive field and thus output spatial dimensions.
    # Values: Single int, 2-tuple (H,W), or 4-tuple. For 2D pooling, typically use (H,W).
    # Discretization: Include boundary (1), typical small values, and a larger value.
    ksize: List[Union[int, Tuple[int, int], List[int]]] = field(
        default_factory=lambda: [1, 2, 3, 5, 7, (1, 1), (2, 2), (3, 3), (4, 4), (1, 3)]
    )
    
    # strides: Step size of sliding window. Directly controls downsampling factor.
    # Values: Single int, 2-tuple (H,W), or 4-tuple.
    # Discretization: Include 1 (no downsampling), typical downsampling values, and a larger stride.
    strides: List[Union[int, Tuple[int, int], List[int]]] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, (1, 1), (2, 2), (3, 3), (1, 2), (2, 1)]
    )
    
    # padding: Controls whether to pad input. 'VALID'=no padding, 'SAME'=pad to maintain spatial size.
    # Also supports explicit padding lists which directly specify padding amounts per spatial dimension.
    # Discretization: Include both string modes and representative explicit padding configurations.
    padding: List[Union[str, List[List[int]]]] = field(
        default_factory=lambda: [
            'VALID',
            'SAME',
            [[0, 0], [1, 1], [1, 1], [0, 0]],  # Symmetric padding
            [[0, 0], [0, 0], [1, 1], [0, 0]],  # NCHW format symmetric
            [[0, 0], [2, 2], [2, 2], [0, 0]],  # Larger padding
            [[0, 0], [0, 1], [1, 0], [0, 0]],  # Asymmetric padding
        ]
    )
    
    # data_format: Determines which dimensions are considered spatial (H,W).
    # 'NHWC': [Batch, Height, Width, Channels]
    # 'NCHW': [Batch, Channels, Height, Width]
    # 'NCHW_VECT_C': Specialized format for int8 data
    # Discrete parameter: List all supported values.
    data_format: List[str] = field(
        default_factory=lambda: ['NHWC', 'NCHW', 'NCHW_VECT_C']
    )