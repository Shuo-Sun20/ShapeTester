import torch
from dataclasses import dataclass, field
from typing import List, Optional

def call_func(inputs, n=None, dim=-1, norm=None, out=None):
    return torch.fft.hfft(input=inputs, n=n, dim=dim, norm=norm, out=out)

n = 10
real_signal = torch.randn(n)
half_hermitian = torch.fft.rfft(real_signal)
example_output = call_func(half_hermitian, n=n)

valid_test_case = {
    'inputs': half_hermitian,
    'n': n,
    'dim': -1,
    'norm': None,
    'out': None
}

@dataclass
class InputSpace:
    n: List[Optional[int]] = field(default_factory=lambda: [None, 4, 5, 6, 8, 10, 12, 16, 20])
    dim: List[int] = field(default_factory=lambda: [-3, -2, -1, 0, 1, 2, 3])
    norm: List[Optional[str]] = field(default_factory=lambda: [None, 'backward', 'forward', 'ortho'])