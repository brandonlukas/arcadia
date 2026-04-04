"""Tests for color quantization module."""

import numpy as np
from PIL import Image

from arcadia.grid import GridResult
from arcadia.palette import quantize, _dominant_color, _merge_similar_colors
from tests.conftest import make_pixel_art


class TestDominantColor:
    def test_uniform_cell(self):
        cell = np.full((16, 16, 3), [100, 150, 200], dtype=np.uint8)
        assert _dominant_color(cell) == (100, 150, 200)

    def test_cell_with_edge_noise(self):
        """Central region color should win over edge noise."""
        cell = np.full((16, 16, 3), [100, 100, 100], dtype=np.uint8)
        # Add different colors at edges
        cell[0, :] = [200, 200, 200]
        cell[-1, :] = [200, 200, 200]
        cell[:, 0] = [200, 200, 200]
        cell[:, -1] = [200, 200, 200]
        r, g, b = _dominant_color(cell)
        assert abs(r - 100) < 20
        assert abs(g - 100) < 20
        assert abs(b - 100) < 20


class TestMergeSimilarColors:
    def test_identical_colors(self):
        colors = [(100, 100, 100), (100, 100, 100)]
        result = _merge_similar_colors(colors)
        assert len(set(result.values())) == 1

    def test_similar_colors_merge(self):
        colors = [(100, 100, 100), (105, 102, 98)]
        result = _merge_similar_colors(colors, threshold=30)
        assert result[(100, 100, 100)] == result[(105, 102, 98)]

    def test_different_colors_stay_separate(self):
        colors = [(0, 0, 0), (255, 255, 255)]
        result = _merge_similar_colors(colors, threshold=30)
        assert result[(0, 0, 0)] != result[(255, 255, 255)]

    def test_four_distinct_colors(self):
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        ]
        result = _merge_similar_colors(colors, threshold=30)
        unique_reps = set(result.values())
        assert len(unique_reps) == 4


class TestQuantize:
    def test_clean_grid_produces_correct_palette_size(self):
        img, palette, indices = make_pixel_art(8, 8, 16, num_colors=4)
        grid = GridResult(cell_size=16, grid_width=8, grid_height=8, confidence=0.9)
        result = quantize(img, grid)
        # Should produce roughly 4 colors (may be fewer if random seed causes duplicates)
        assert len(result.palette) <= 6
        assert len(result.palette) >= 2

    def test_color_grid_dimensions(self):
        img, _, _ = make_pixel_art(16, 16, 8)
        grid = GridResult(cell_size=8, grid_width=16, grid_height=16, confidence=0.9)
        result = quantize(img, grid)
        assert len(result.color_grid) == 16
        assert all(len(row) == 16 for row in result.color_grid)

    def test_hex_format(self):
        img, _, _ = make_pixel_art(4, 4, 32, num_colors=2)
        grid = GridResult(cell_size=32, grid_width=4, grid_height=4, confidence=0.9)
        result = quantize(img, grid)
        for row in result.color_grid:
            for color in row:
                assert color.startswith("#")
                assert len(color) == 7

    def test_antialiased_produces_reasonable_palette(self):
        img, _, _ = make_pixel_art(16, 16, 16, num_colors=4, antialiased=True)
        grid = GridResult(cell_size=16, grid_width=16, grid_height=16, confidence=0.9)
        result = quantize(img, grid)
        # Anti-aliased image should still quantize to a small palette
        assert len(result.palette) <= 8
