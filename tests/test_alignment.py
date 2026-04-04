"""Tests for local grid alignment module."""

import numpy as np
from PIL import Image
import pytest

from arcadia.alignment import align_grid, AlignedGrid, _compute_edge_strength, _find_boundaries
from arcadia.grid import detect_grid, GridResult
from tests.conftest import make_pixel_art, make_drifted_pixel_art


class TestComputeEdgeStrength:
    def test_uniform_image_has_no_edges(self):
        gray = np.full((64, 64), 128.0)
        h_strength = _compute_edge_strength(gray, axis=1)
        assert h_strength.max() < 1.0

    def test_vertical_stripe_has_horizontal_edges(self):
        gray = np.zeros((64, 64))
        gray[:, 32:] = 255.0
        h_strength = _compute_edge_strength(gray, axis=1)
        # Strong edge at column 31-32 boundary
        assert h_strength[31] > h_strength[0]
        assert h_strength[31] > h_strength[50]


class TestFindBoundaries:
    def test_perfect_grid_returns_evenly_spaced(self):
        # Create a simple edge signal with peaks every 16px
        edge = np.zeros(255)
        for i in range(15, 255, 16):
            edge[i] = 100.0
        boundaries = _find_boundaries(edge, cell_size=16, dim=256)
        assert boundaries[0] == 0
        assert boundaries[-1] == 256
        assert len(boundaries) == 17  # 16 cells + 1

    def test_drifted_edges_snap_correctly(self):
        # Create edge signal with peaks slightly off from ideal
        edge = np.zeros(255)
        # Ideal boundaries at 16, 32, 48, ... but drift by 1-2px
        for i, pos in enumerate([14, 31, 49, 63, 80, 97, 111, 128,
                                  143, 160, 177, 191, 208, 225, 239]):
            edge[pos] = 100.0
        boundaries = _find_boundaries(edge, cell_size=16, dim=256)
        assert boundaries[0] == 0
        assert boundaries[-1] == 256
        # Boundaries should snap to the actual edges, not ideal positions
        assert boundaries[1] == 15  # snapped to edge at 14 (+1 for boundary pos)

    def test_monotonically_increasing(self):
        rng = np.random.RandomState(42)
        edge = rng.random(255) * 50
        boundaries = _find_boundaries(edge, cell_size=16, dim=256)
        for i in range(1, len(boundaries)):
            assert boundaries[i] > boundaries[i - 1]


class TestAlignGrid:
    def test_perfect_grid_alignment_close_to_fixed(self):
        """On a perfect grid, alignment should produce boundaries near the ideal positions."""
        img, _, _ = make_pixel_art(16, 16, 16)
        grid = GridResult(cell_size=16, grid_width=16, grid_height=16, confidence=0.9)
        aligned = align_grid(img, grid)
        assert isinstance(aligned, AlignedGrid)
        assert aligned.grid_width == 16
        assert aligned.grid_height == 16
        assert len(aligned.x_boundaries) == 17
        assert len(aligned.y_boundaries) == 17
        # Boundaries should be close to multiples of 16
        for i, b in enumerate(aligned.x_boundaries):
            assert abs(b - i * 16) <= 2

    def test_drifted_grid_boundaries_snap_to_edges(self, drifted_grid):
        """On a drifted grid, alignment should find the actual boundaries."""
        img, gw, gh, cs, true_x, true_y = drifted_grid
        # Use known grid params directly — we're testing alignment, not detection
        grid = GridResult(cell_size=cs, grid_width=gw, grid_height=gh, confidence=0.9)
        aligned = align_grid(img, grid)
        assert aligned.grid_width == gw
        assert aligned.grid_height == gh
        # Aligned boundaries should be closer to true boundaries than fixed grid
        fixed_x = [i * cs for i in range(gw + 1)]
        aligned_error = sum(abs(a - t) for a, t in zip(aligned.x_boundaries, true_x))
        fixed_error = sum(abs(f - t) for f, t in zip(fixed_x, true_x))
        assert aligned_error <= fixed_error

    def test_e2e_drifted_pipeline(self, tmp_path, drifted_grid):
        """Full pipeline works with drifted pixel art via manual grid + alignment."""
        from arcadia.palette import quantize
        from arcadia.svg import render
        img, gw, gh, cs, _, _ = drifted_grid
        grid = GridResult(cell_size=cs, grid_width=gw, grid_height=gh, confidence=0.9)
        aligned = align_grid(img, grid)
        palette_result = quantize(img, aligned)
        svg = render(palette_result, aligned)
        assert 'shape-rendering="crispEdges"' in svg
        assert f'viewBox="0 0 {gw} {gh}"' in svg

    def test_alignment_is_fast(self):
        """Alignment step should be very fast (< 0.1s for 1024x768)."""
        import time
        img, _, _ = make_pixel_art(64, 48, 16)
        grid = GridResult(cell_size=16, grid_width=64, grid_height=48, confidence=0.9)
        t0 = time.perf_counter()
        aligned = align_grid(img, grid)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.1
