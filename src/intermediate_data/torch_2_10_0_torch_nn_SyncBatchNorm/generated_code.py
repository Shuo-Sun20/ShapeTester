import torch

def call_func(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, process_group=None, inputs=None):
    sync_bn = torch.nn.SyncBatchNorm(num_features, eps, momentum, affine, track_running_stats, process_group)
    return sync_bn(inputs)

example_input = torch.randn(2, 3, 4, 4)
example_output = call_func(num_features=3, inputs=example_input)