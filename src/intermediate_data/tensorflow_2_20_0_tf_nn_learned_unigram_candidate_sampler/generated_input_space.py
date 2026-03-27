import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Any

# 1. Define valid_test_case dictionary
valid_test_case = {
    'inputs': [tf.constant([[1, 3], [5, 7], [2, 4], [6, 8]], dtype=tf.int64)],
    'num_true': 2,
    'num_sampled': 3,
    'unique': True,
    'range_max': 10,
    'seed': 0,
    'name': None
}

# 2. & 3. Parameters affecting output shape (excluding "inputs") and their value spaces
# num_true: int, affects shape of true_expected_count [batch_size, num_true]
num_true_values = [1, 2, 3, 5, 10]  # boundary and typical values

# num_sampled: int, affects shape of sampled_candidates [num_sampled] and sampled_expected_count [num_sampled]
num_sampled_values = [1, 3, 10, 50, 100]  # boundary and typical values

# batch_size: derived from input tensor, but not a direct parameter of call_func
# range_max: int, affects sampling range but not output shapes
# unique: bool, affects sampling behavior but not output shapes
# seed: int, affects randomness but not output shapes
# name: str, affects operation name but not output shapes

@dataclass
class InputSpace:
    """
    Dataclass containing parameters that affect output tensor shapes for
    tf.nn.learned_unigram_candidate_sampler.
    """
    num_true: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    num_sampled: List[int] = field(default_factory=lambda: [1, 3, 10, 50, 100])

# Example instantiation
var = InputSpace()