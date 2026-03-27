import torch
import torch.nn.functional as F

def call_func(inputs, weight, offsets=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, mode='mean', sparse=False, per_sample_weights=None, include_last_offset=False, padding_idx=None):
    input_tensor = inputs
    return F.embedding_bag(input_tensor, weight, offsets, max_norm, norm_type, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx)

torch.manual_seed(42)
input_tensor = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.long)
weight = torch.randn(10, 3, requires_grad=True)
offsets = torch.tensor([0, 4], dtype=torch.long)
example_output = call_func(input_tensor, weight, offsets)