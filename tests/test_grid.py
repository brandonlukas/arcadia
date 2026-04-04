"""Tests for grid detection module."""

import numpy as np
from PIL import Image
import pytest

from arcadia.grid import detect_grid, GridDetectionError
from tests.conftest import make_pixel_art


class TestDetectGrid:
    def test_8x8_grid(self, small_grid):
        img, gw, gh, cs, _, _ = small_grid
        result = detect_grid(img)
        assert result.cell_size == cs
        assert result.grid_width == gw
        assert result.grid_height == gh
        assert result.confidence >= 0.5

    def test_32x32_grid(self, medium_grid):
        img, gw, gh, cs, _, _ = medium_grid
        result = detect_grid(img)
        assert result.cell_size == cs
        assert result.grid_width == gw
        assert result.grid_height == gh

    def test_non_power_of_2_24x24(self, non_power_of_2_grid_24):
        img, gw, gh, cs, _, _ = non_power_of_2_grid_24
        result = detect_grid(img)
        assert result.cell_size == cs
        assert result.grid_width == gw
        assert result.grid_height == gh

    def test_non_power_of_2_20x20(self, non_power_of_2_grid_20):
        img, gw, gh, cs, _, _ = non_power_of_2_grid_20
        result = detect_grid(img)
        assert result.cell_size == cs
        assert result.grid_width == gw
        assert result.grid_height == gh

    def test_antialiased_grid(self, antialiased_grid):
        img, gw, gh, cs, _, _ = antialiased_grid
        result = detect_grid(img)
        assert result.cell_size == cs
        assert result.grid_width == gw
        assert result.grid_height == gh

    def test_rgba_input(self, rgba_grid):
        img, gw, gh, cs, _, _ = rgba_grid
        # RGBA images should be converted to RGB before calling detect_grid
        rgb_img = img.convert("RGB")
        result = detect_grid(rgb_img)
        assert result.cell_size == cs
        assert result.grid_width == gw
        assert result.grid_height == gh

    def test_64x64_grid(self):
        img, _, _ = make_pixel_art(64, 64, 8)
        result = detect_grid(img)
        assert result.cell_size == 8
        assert result.grid_width == 64
        assert result.grid_height == 64

    def test_16x16_grid(self):
        img, _, _ = make_pixel_art(16, 16, 16)
        result = detect_grid(img)
        assert result.cell_size == 16
        assert result.grid_width == 16
        assert result.grid_height == 16

    def test_uniform_image_raises(self):
        """A solid-color image should fail grid detection."""
        img = Image.new("RGB", (256, 256), (128, 128, 128))
        with pytest.raises(GridDetectionError, match="uniform"):
            detect_grid(img)

    def test_photograph_raises(self):
        """A noisy non-pixel-art image should fail or have low confidence."""
        rng = np.random.RandomState(99)
        noise = rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(noise, "RGB")
        with pytest.raises(GridDetectionError):
            detect_grid(img)

    def test_confidence_higher_for_clean_grid(self):
        clean_img, _, _ = make_pixel_art(32, 32, 16, antialiased=False)
        aa_img, _, _ = make_pixel_art(32, 32, 16, antialiased=True)
        clean_result = detect_grid(clean_img)
        aa_result = detect_grid(aa_img)
        assert clean_result.confidence >= aa_result.confidence
