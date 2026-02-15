import warnings
warnings.simplefilter("ignore")
import torch
import numpy as np


class CapacityConstrainedMemoryBuffer:
    def __init__(self, 
        memory_capacity: int, 
        alpha: float = 0.02,
        seed: int = 0
    ):
        """
        memory_capacity (int): capacity C (in bits) of the storge. It models the total number of functional binary 
            synaptic slots. Larger C permits longer codes per item, reducing overlap between codes.
        alpha (float): noise scaling representing synaptic unreliability. Larger alphas yield bit-flip probabilities 
            for a given memory load, as a function of the number of item in the storage.
        seed: (int) = random seed for proejction matrix generation 
        """
        self.memory_capacity = memory_capacity
        self.alpha = alpha
        self.seed = seed
        self.memory_buffer = None

    def _make_R(self, D, m):
        rng = np.random.default_rng(self.seed)
        # Random Gaussian projection (unit variance), scaled for stability
        return rng.standard_normal((D, m)) / np.sqrt(D)
    
    def encode(self, keys: np.ndarray):
        """
        Create the capacity-constrained binary hippocampal buffer to stoer compressed keys.

        keys (np.ndarray): tensor of shape (N, D) indicating N D_dimenstional items (embeddings) to
            store in the memory buffer
        """
        # Each stored bit is flipped with probability p = min(0.5, alpha * N / max(1, m)).
        N, D = keys.shape
        m = max(1, self.memory_capacity // N) # bits per item set by capacity 
        R = self._make_R(D, m)
        self.R = R
        proj = keys @ R                  # (N, m)
        codes = (proj >= 0)              # boolean codes

        # Capacity-driven storage noise (biologically plausible synaptic unreliability)
        p = min(0.5, self.alpha * N / max(1, m))
        if p > 0:
            rng = np.random.default_rng(self.seed + 1)
            flips = rng.uniform(size=codes.shape) < p
            codes = np.logical_xor(codes, flips)

        self.memory_buffer = codes
    
    def get_distances(self, query: np.ndarray):
        """
        Compute Hamming distance for query-key matching

        query (np.ndarray): tensor of shape (n, D) to use as query to compute query-key distances
        """
        if self.memory_buffer is None:
            raise AttributeError("Memory Buffer still not created...")

        q = (query @ self.R) >= 0

        # Hamming distance = number of bit flips
        # XOR then count True
        d = np.count_nonzero(self.memory_buffer ^ q, axis=1)
        return d