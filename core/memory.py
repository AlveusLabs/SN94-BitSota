import numpy as np


class Memory:
    """memory layout for automl-zero style execution"""

    def __init__(self, input_dim: int):
        self.input_dim = input_dim

        # memory banks - bigger to allow more complex algorithms
        self.scalars = np.zeros(20, dtype=np.float32)  # s0-s19
        self.vectors = np.zeros(
            (10, input_dim + 10), dtype=np.float32
        )  # v0-v9, extra space
        self.matrices = np.zeros(
            (5, input_dim + 10, input_dim + 10), dtype=np.float32
        )  # m0-m4

        # special locations by convention
        # s0 = prediction, s1 = label (when loaded)
        # v0 = input features (when loaded)

    def reset_per_sample(self):
        """reset transient values between samples but keep learned params"""
        # only clear certain locations, not all memory
        self.scalars[0:2] = 0  # clear prediction and label slots
        self.vectors[0] = 0  # clear input slot
