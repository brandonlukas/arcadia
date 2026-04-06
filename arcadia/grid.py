"""Grid detection via autocorrelation of gradient signal for pixel art images."""

from dataclasses import dataclass

import numpy as np
from PIL import Image


class GridDetectionError(Exception):
    """Raised when the pixel grid cannot be reliably detected."""


@dataclass
class GridResult:
    cell_size: int
    grid_width: int
    grid_height: int
    confidence: float


def _autocorrelation(signal: np.ndarray) -> np.ndarray:
    """Compute normalized autocorrelation of a 1D signal."""
    n = len(signal)
    mean = signal.mean()
    centered = signal - mean
    variance = np.sum(centered ** 2)
    if variance < 1e-10:
        return np.zeros(n)
    result = np.correlate(centered, centered, mode="full")
    result = result[n - 1:]  # keep only non-negative lags
    return result / variance


def _gradient_autocorrelation(line: np.ndarray) -> np.ndarray:
    """Compute autocorrelation of the absolute gradient of a 1D signal.

    For pixel art, raw pixels have random colors per cell (no periodicity),
    but the gradient (transitions between cells) has spikes at every cell
    boundary, which ARE periodic with period = cell_size.
    """
    grad = np.abs(np.diff(line))
    return _autocorrelation(grad)


def _find_peaks_with_prominence(
    autocorr: np.ndarray,
    min_lag: int = 4,
    min_prominence: float = 0.05,
) -> list[tuple[int, float]]:
    """Find peaks in autocorrelation with prominence filtering.

    Returns list of (lag, prominence) sorted by lag.
    """
    peaks = []
    n = len(autocorr)
    if n < min_lag + 2:
        return peaks

    for i in range(max(min_lag, 1), n - 1):
        if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
            # Compute prominence: height above the higher of the two nearest valleys
            # Look left to find the valley between this peak and the previous one
            left_start = max(1, i - i // 2)  # search within half a period
            left_min = np.min(autocorr[left_start:i])
            # Look right for the next valley
            right_end = min(n, i + i // 2 + 1)
            right_min = np.min(autocorr[i + 1:right_end]) if i + 1 < right_end else autocorr[i]
            prominence = autocorr[i] - max(left_min, right_min)
            if prominence >= min_prominence:
                peaks.append((i, prominence))

    return peaks


def _best_cell_size(
    peaks: list[tuple[int, float]], image_dim: int,
) -> tuple[int, float] | None:
    """Select the best cell size from peaks.

    Prefers the smallest lag that evenly divides the image dimension,
    as long as it has reasonable prominence. In pixel art, the fundamental
    frequency (smallest cell size) is correct — larger peaks are harmonics.
    """
    candidates = []
    for tolerance in (0.02, 0.05):
        for lag, prominence in peaks:
            if lag <= 0 or lag > image_dim // 2:
                continue
            ratio = image_dim / lag
            rounded = round(ratio)
            if rounded < 2:
                continue
            remainder = abs(ratio - rounded) / rounded
            if remainder <= tolerance:
                candidates.append((lag, prominence, remainder))
        if candidates:
            break

    if not candidates:
        return None

    # Filter to candidates with at least 30% of the max prominence
    # (avoids picking noise peaks over real harmonic peaks)
    max_prom = max(c[1] for c in candidates)
    viable = [c for c in candidates if c[1] >= max_prom * 0.3]

    # Among viable candidates, prefer the smallest lag (fundamental frequency).
    # In pixel art, smaller cell = finer grid = correct detection.
    # Larger peaks are harmonics (2x, 3x the real cell size).
    viable.sort(key=lambda c: c[0])  # sort by lag ascending
    best_lag, best_prominence, _ = viable[0]
    return best_lag, best_prominence


def _collect_candidates(
    peaks: list[tuple[int, float]], image_dim: int,
) -> list[tuple[int, float]]:
    """Collect all viable cell size candidates from peaks."""
    candidates = []
    for tolerance in (0.02, 0.05):
        for lag, prominence in peaks:
            if lag <= 0 or lag > image_dim // 2:
                continue
            ratio = image_dim / lag
            rounded = round(ratio)
            if rounded < 2:
                continue
            remainder = abs(ratio - rounded) / rounded
            if remainder <= tolerance:
                candidates.append((lag, prominence))
        if candidates:
            break
    return candidates


def detect_grid(image: Image.Image) -> GridResult:
    """Detect the pixel grid in an AI-generated pixel art image.

    Uses autocorrelation of the gradient signal, then scores the top candidates
    by cell uniformity to avoid picking sub-cell texture as the grid.

    Args:
        image: RGB PIL Image.

    Returns:
        GridResult with detected grid parameters.

    Raises:
        GridDetectionError: If grid cannot be reliably detected.
    """
    gray = np.array(image.convert("L"), dtype=np.float64)
    pixels = np.array(image)
    height, width = gray.shape

    n_row_samples = max(20, height // 10)
    n_col_samples = max(20, width // 10)

    row_indices = np.linspace(0, height - 1, n_row_samples, dtype=int)
    col_indices = np.linspace(0, width - 1, n_col_samples, dtype=int)

    row_lines = gray[row_indices, :]
    col_lines = gray[:, col_indices].T

    def _filter_low_variance(lines: np.ndarray) -> np.ndarray:
        variances = np.var(lines, axis=1)
        max_var = variances.max()
        if max_var < 1e-10:
            return lines[:0]
        return lines[variances >= max_var * 0.05]

    row_filtered = _filter_low_variance(row_lines)
    col_filtered = _filter_low_variance(col_lines)

    total_filtered = len(row_filtered) + len(col_filtered)
    if total_filtered == 0:
        raise GridDetectionError(
            "Image appears to be uniform — no pixel grid detected. "
            "This may not be a pixel art image."
        )
    if total_filtered < 3:
        raise GridDetectionError(
            "Too few high-contrast lines found for reliable grid detection. "
            "This may not be a pixel art image."
        )

    max_lag_h = width // 2
    max_lag_v = height // 2

    def _avg_gradient_autocorr(lines: np.ndarray, max_lag: int) -> np.ndarray:
        if len(lines) == 0:
            return np.zeros(max_lag)
        accum = np.zeros(max_lag)
        count = 0
        for line in lines:
            ac = _gradient_autocorrelation(line)
            length = min(len(ac), max_lag)
            accum[:length] += ac[:length]
            count += 1
        return accum / max(count, 1)

    avg_h = _avg_gradient_autocorr(row_filtered, max_lag_h)
    avg_v = _avg_gradient_autocorr(col_filtered, max_lag_v)

    peaks_h = _find_peaks_with_prominence(avg_h)
    peaks_v = _find_peaks_with_prominence(avg_v)

    # Collect autocorrelation candidates from both axes
    cands_h = _collect_candidates(peaks_h, width)
    cands_v = _collect_candidates(peaks_v, height)

    if not cands_h and not cands_v:
        raise GridDetectionError(
            "No periodic grid pattern detected in the image. "
            "The autocorrelation found no significant peaks. "
            "This may not be a pixel art image."
        )

    # Gather unique cell sizes with their best prominence
    cell_size_proms: dict[int, float] = {}
    for lag, prom in cands_h + cands_v:
        if lag not in cell_size_proms or prom > cell_size_proms[lag]:
            cell_size_proms[lag] = prom

    max_prom = max(cell_size_proms.values())
    viable = {cs: p for cs, p in cell_size_proms.items() if p >= max_prom * 0.3}

    # Among viable candidates, prefer the smallest (fundamental frequency)
    sorted_viable = sorted(viable.keys())
    cell_size = sorted_viable[0]
    confidence = min(1.0, viable[cell_size] * 4.0)

    confidence = min(1.0, max(0.0, confidence))

    grid_width = round(width / cell_size)
    grid_height = round(height / cell_size)

    if confidence < 0.5:
        raise GridDetectionError(
            f"Grid detection confidence too low ({confidence:.2f}). "
            f"Best candidate: cell_size={cell_size}, grid={grid_width}x{grid_height}. "
            "This may not be a pixel art image or the grid pattern is too weak."
        )

    return GridResult(
        cell_size=cell_size,
        grid_width=grid_width,
        grid_height=grid_height,
        confidence=confidence,
    )
