import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Any

# 1. Define valid_test_case
batch_size = 2
target_seq_len = 10
source_seq_len = 15
head_dim = 64
num_query_heads = 8
num_key_value_heads = 2

# All input tensors must have the same last dimension (feature_dim)
feature_dim = 512  # Common feature dimension for all inputs

np.random.seed(42)
query = np.random.randn(batch_size, target_seq_len, feature_dim).astype(np.float32)
value = np.random.randn(batch_size, source_seq_len, feature_dim).astype(np.float32)
key = np.random.randn(batch_size, source_seq_len, feature_dim).astype(np.float32)

valid_test_case = {
    'head_dim': head_dim,
    'num_query_heads': num_query_heads,
    'num_key_value_heads': num_key_value_heads,
    'inputs': [query, value, key],
    'dropout': 0.1,
    'use_bias': True,
    'flash_attention': None,
    'kernel_initializer': "glorot_uniform",
    'bias_initializer': "zeros",
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None,
    'seed': None,
    'attention_mask': None,
    'return_attention_scores': False,
    'training': None,
    'use_causal_mask': False
}

# 2. Parameters affecting output tensor shape (except "inputs"):
# Only "num_query_heads" affects the attention_scores shape when return_attention_scores=True
# The attention_output shape is always determined by the input query shape

# 3. Parameter types and discretized value spaces
# num_query_heads: discrete positive integer
# return_attention_scores: discrete boolean

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    num_query_heads: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    return_attention_scores: List[bool] = field(default_factory=lambda: [True, False])