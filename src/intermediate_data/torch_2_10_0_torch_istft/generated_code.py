import torch

def call_func(
    inputs: torch.Tensor,
    n_fft: int,
    hop_length: int = None,
    win_length: int = None,
    window: torch.Tensor = None,
    center: bool = True,
    normalized: bool = False,
    onesided: bool = None,
    length: int = None,
    return_complex: bool = False
) -> torch.Tensor:
    return torch.istft(
        input=inputs,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        normalized=normalized,
        onesided=onesided,
        length=length,
        return_complex=return_complex
    )

n_fft = 512
hop_length = 256
win_length = 512
window = torch.hann_window(win_length)
batch_size = 2
T = 10
N = n_fft // 2 + 1

inputs = torch.randn(batch_size, N, T, dtype=torch.complex64)
example_output = call_func(
    inputs=inputs,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window=window,
    center=True,
    normalized=False,
    onesided=True,
    length=None,
    return_complex=False
)