"""
Path embeddings shared by the Chapter 5 signature experiments.

Single source of truth for the path transformations used by 5_1_2, 5_2_1
and 5_3_1. Do not redefine them locally in the scripts.

Every transform maps a raw (T, d) series to a piecewise-linear path whose
first channel is a normalized time t ∈ [0, 1], added exactly once:

    lin+time           : (T, d)   → (T, d+1)       (t, x)
    lin+time+lead-lag  : (T, d)   → (T+1, 2d+1)    (t, lead, lag)
    rectilinear+time   : (T, d)   → ((T-1)(d+1)+1, d+1)  values one by one, time last
    lin+time+cumsum    : (T, d)   → (T, d+1)       (t, cumsum(x))

Scripts import this module with `from embeddings import ...`; this works
because `python experiments/chap5/<script>.py` puts the script's directory
on sys.path.
"""

from typing import Callable, Dict

import numpy as np


def _time_channel(T: int) -> np.ndarray:
    """Normalized time t ∈ [0, 1] as a (T, 1) column."""
    return np.linspace(0.0, 1.0, T, dtype=np.float64).reshape(-1, 1)


def lin_time(x: np.ndarray) -> np.ndarray:
    """Prepend a normalized-time channel: (T, d) → (T, d+1)."""
    x = np.asarray(x, dtype=np.float64)
    return np.hstack([_time_channel(x.shape[0]), x])


def lin_time_lead_lag(x: np.ndarray) -> np.ndarray:
    """Delayed-copy lead-lag (Fermanian, 2021, eq. 9), lag of one step: (T, d) → (T+1, 2d+1).

        lead = [x_1, ..., x_T, x_T]   (last point repeated)
        lag  = [0,   x_1, ..., x_T]   (zero before the first step)

    Lead and lag move simultaneously, the lag being the lead delayed by one
    step. Time is added once, uniform on [0, 1] over the T+1 points. This is
    NOT the staircase lead-lag (Chevyrev–Kormilitzin / Flint et al.), whose
    lead and lag channels move alternately. Not translation-invariant: the
    zero start of the lag encodes the level of the series.
    """
    x = np.asarray(x, dtype=np.float64)
    lead = np.vstack([x, x[-1:]])
    lag = np.vstack([np.zeros_like(x[:1]), x])
    return np.hstack([_time_channel(lead.shape[0]), lead, lag])


def rectilinear_time(x: np.ndarray) -> np.ndarray:
    """Rectilinear time-augmented path (Fermanian, 2021): (T, d) → ((T-1)(d+1)+1, d+1).

    Matches `time_rectilinear` in github.com/afermanian/embedding_with_signatures:
    each transition is split into d+1 axis-parallel segments, the value
    channels moving one at a time in channel order, then time last. Time is
    stored first in the output, like the other configurations (channel order
    only permutes signature features).
    """
    x = np.asarray(x, dtype=np.float64)
    path = np.hstack([x, _time_channel(x.shape[0])])   # time last = move order
    T, D = path.shape
    out = np.empty(((T - 1) * D + 1, D), dtype=np.float64)
    out[0] = path[0]
    for j in range(1, D + 1):
        # j-th point of each transition: first j channels already moved.
        out[j::D, :j] = path[1:, :j]
        out[j::D, j:] = path[:-1, j:]
    return np.hstack([out[:, -1:], out[:, :-1]])


def lin_time_cumsum(x: np.ndarray) -> np.ndarray:
    """Cumulative sum on all raw channels, then time-augment: (T, d) → (T, d+1).

    Applied to every channel, including the CharTraj pressure channel.
    """
    return lin_time(np.cumsum(np.asarray(x, dtype=np.float64), axis=0))


TRANSFORMS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "lin+time":          lin_time,
    "lin+time+lead-lag": lin_time_lead_lag,
    "rectilinear+time":  rectilinear_time,
    "lin+time+cumsum":   lin_time_cumsum,
}


def path_dim(config: str, d: int) -> int:
    """Channel count of the embedded path for a raw series with d channels."""
    if config == "lin+time+lead-lag":
        return 2 * d + 1
    if config in TRANSFORMS:
        return d + 1
    raise KeyError(f"Unknown path configuration: {config!r}")
