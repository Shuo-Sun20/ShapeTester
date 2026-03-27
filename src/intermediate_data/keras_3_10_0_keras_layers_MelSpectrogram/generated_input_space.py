import keras
from dataclasses import dataclass
from typing import List, Optional, Union

# 1. Define valid_test_case with all parameters
input_tensor = keras.random.uniform(shape=(2, 16000))
valid_test_case = {
    "inputs": input_tensor,
    "fft_length": 2048,
    "sequence_stride": 512,
    "sequence_length": None,
    "window": "hann",
    "sampling_rate": 16000,
    "num_mel_bins": 128,
    "min_freq": 20.0,
    "max_freq": None,
    "power_to_db": True,
    "top_db": 80.0,
    "mag_exp": 2.0,
    "min_power": 1e-10,
    "ref_power": 1.0
}

# 2. Parameters affecting output shape: fft_length, sequence_stride, sequence_length, num_mel_bins
# 3. & 4. Define InputSpace dataclass with discretized value spaces
@dataclass
class InputSpace:
    fft_length: List[int] = None
    sequence_stride: List[int] = None
    sequence_length: List[Optional[int]] = None
    num_mel_bins: List[int] = None
    
    def __post_init__(self):
        if self.fft_length is None:
            self.fft_length = [256, 512, 1024, 2048, 4096]  # Powers of 2, typical values
        if self.sequence_stride is None:
            self.sequence_stride = [64, 128, 256, 512, 1024]  # Common hop lengths
        if self.sequence_length is None:
            self.sequence_length = [None, 256, 512, 1024, 2048]  # 5 values including None
        if self.num_mel_bins is None:
            self.num_mel_bins = [40, 64, 80, 128, 256]  # Common mel bin counts