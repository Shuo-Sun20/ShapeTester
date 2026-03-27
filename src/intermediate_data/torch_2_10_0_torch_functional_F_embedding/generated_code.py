import torch
import torch.nn.functional as F

def call_func(inputs, weight, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False):
    input_tensor = inputs
    return F.embedding(input_tensor, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

torch.manual_seed(42)
example_input = torch.randint(0, 10, (2, 4))
example_weight = torch.randn(10, 3)
example_output = call_func(example_input, example_weight, padding_idx=0)