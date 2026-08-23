"""Channel-sampling preprocessing strategies for turning a stack of grayscale
video frames into a 3-channel "RGB-shaped" input for a standard 2D CNN.

All transforms share the same signature:

    transform(frames: (N, H, W) array) -> (3, H, W) array

where `frames` is a window of N consecutive grayscale frames (values expected
in [0, 1]) and the output is a 3-channel image ready to feed a normal 2D CNN.

Works with either numpy arrays or torch tensors (numpy in, numpy out; torch
in, torch out) since the ops used (indexing, matmul) are supported by both.
"""
import numpy as np


def _n_frames(frames):
    return frames.shape[0]


def grayst(frames):
    """GrayST (Kim et al. 2022): stack 3 consecutive raw grayscale frames as
    the R, G, B channels. Uses the last 3 frames of the given window (the
    most recent consecutive triplet), matching the original paper's use of
    3 back-to-back frames.
    """
    n = _n_frames(frames)
    if n < 3:
        raise ValueError(f"grayst needs at least 3 frames, got {n}")
    return frames[n - 3:n]


def tc_reordering(frames):
    """TC (Temporal-Channel) Reordering: sample 3 frames spread across the
    *whole* window (start, middle, end) and stack their raw pixel values as
    channels. This gives a wider temporal footprint than GrayST but, being a
    raw pixel stack, is directly sensitive to any global pixel shift
    (e.g. camera pan) the same way it is to local object motion.
    """
    n = _n_frames(frames)
    if n < 3:
        raise ValueError(f"tc_reordering needs at least 3 frames, got {n}")
    idx = [0, n // 2, n - 1]
    return frames[idx]


def chunk_avg(frames, n_chunks=3):
    """Chunk-average control: split the window into `n_chunks` contiguous,
    (near-)equal groups and average each group into one channel. Unlike
    GrayST/TC Reordering, every single frame in the window contributes to
    exactly one output channel -- nothing is discarded. Unlike FreqST, there
    is no frequency decomposition: each channel is just a coarse local
    average, so it isolates "uses 100% of the window" from "uses a DCT."
    """
    n = _n_frames(frames)
    if n < n_chunks:
        raise ValueError(f"chunk_avg needs at least {n_chunks} frames, got {n}")
    idx_groups = np.array_split(np.arange(n), n_chunks)
    return np.stack([frames[idx].mean(axis=0) for idx in idx_groups], axis=0)


def _dct_matrix(n, dtype=np.float64):
    """Orthonormal DCT-II basis matrix of shape (n, n).

    Row k gives the weights to project a length-n signal onto DCT basis
    function k. Row 0 is the (constant) DC component.
    """
    k = np.arange(n).reshape(-1, 1)
    t = np.arange(n).reshape(1, -1)
    basis = np.cos(np.pi / n * (t + 0.5) * k)
    alpha = np.full((n, 1), np.sqrt(2.0 / n))
    alpha[0, 0] = np.sqrt(1.0 / n)
    return (alpha * basis).astype(dtype)


def freqst(frames, num_coeffs=3):
    """FreqST (proposed): 1D DCT along the time axis, independently per
    pixel, keeping the first `num_coeffs` coefficients as output channels.

    Channel 0 = DC = average appearance over the window.
    Channel 1 = lowest-frequency AC = slow motion energy.
    Channel 2 = next AC = faster motion energy.

    Implemented as a single matmul against a fixed DCT basis matrix, so it's
    the same computational order as GrayST's grayscale conversion.
    """
    n = _n_frames(frames)
    if n < num_coeffs:
        raise ValueError(f"freqst needs at least {num_coeffs} frames, got {n}")

    is_torch = hasattr(frames, "detach")
    h, w = frames.shape[1], frames.shape[2]
    flat = frames.reshape(n, h * w)

    if is_torch:
        import torch
        basis = torch.as_tensor(_dct_matrix(n), dtype=frames.dtype, device=frames.device)
        coeffs = basis[:num_coeffs] @ flat
        return coeffs.reshape(num_coeffs, h, w)
    else:
        basis = _dct_matrix(n, dtype=np.float64)
        coeffs = basis[:num_coeffs] @ flat.astype(np.float64)
        return coeffs.reshape(num_coeffs, h, w).astype(frames.dtype)


TRANSFORMS = {
    "grayst": grayst,
    "tc_reordering": tc_reordering,
    "freqst": freqst,
}


def apply_windowed(video, transform, window, stride=1):
    """Slide a `window`-length window over a (T, H, W) video and apply
    `transform` to each window, returning a (T', 3, H, W) stack where
    T' = (T - window) // stride + 1.
    """
    t = video.shape[0]
    n_windows = (t - window) // stride + 1
    outs = []
    for i in range(n_windows):
        start = i * stride
        outs.append(transform(video[start:start + window]))
    if hasattr(video, "detach"):
        import torch
        return torch.stack(outs, dim=0)
    return np.stack(outs, axis=0)
