import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, List

valid_test_case = {
    "inputs": tf.random.normal(shape=(5, 3, 3), dtype=tf.float32),
    "name": None
}

@dataclass
class InputSpace:
    """
    Dataclass containing all parameters affecting the shape of the output tensor,
    except for 'inputs'. For tf.linalg.slogdet, only 'name' affects operation
    naming but not tensor shape, so no shape-affecting parameters exist.
    """
    # No parameters in call_func() affect output shape beyond 'inputs'
    pass