"""Shared test fixtures for Arcadia tests."""

import numpy as np
from PIL import Image
import pytest


def make_pixel_art(
    grid_width: int,
    grid_height: int,
    cell_size: int,
    num_colors: int = 4,
    antialiased: bool = False,
    rgba: bool = False,
    seed: int = 42,
) -> Image.Image:
    """Generate a synthetic pixel art image with a known grid.

    Args:
        grid_width: Number of logical pixels wide.
        grid_height: Number of logical pixels tall.
        cell_size: Size of each logical pixel in actual pixels.
        num_colors: Number of distinct colors in the palette.
        antialiased: If True, add anti-aliasing at cell boundaries.
        rgba: If True, return an RGBA image with full opacity.
        seed: Random seed for reproducibility.
    """
    rng = np.random.RandomState(seed)
    # Generate a random palette
    palette = rng.randint(30, 226, size=(num_colors, 3), dtype=np.uint8)

    # Assign each grid cell a random color index
    color_indices = rng.randint(0, num_colors, size=(grid_height, grid_width))

    # Build the full image
    img_h = grid_height * cell_size
    img_w = grid_width * cell_size
    pixels = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    for row in range(grid_height):
        for col in range(grid_width):
            color = palette[color_indices[row, col]]
            y0, y1 = row * cell_size, (row + 1) * cell_size
            x0, x1 = col * cell_size, (col + 1) * cell_size
            pixels[y0:y1, x0:x1] = color

    if antialiased:
        # Simulate anti-aliasing by blending a 1-pixel border between cells
        from PIL import ImageFilter
        img = Image.fromarray(pixels, "RGB")
        # Apply slight blur to simulate AA
        blurred = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        pixels = np.array(blurred)

    img = Image.fromarray(pixels, "RGB")
    if rgba:
        img = img.convert("RGBA")

    return img, palette, color_indices


def make_drifted_pixel_art(
    grid_width: int,
    grid_height: int,
    cell_size: int,
    max_drift: int = 2,
    num_colors: int = 4,
    seed: int = 42,
) -> tuple[Image.Image, list[int], list[int]]:
    """Generate pixel art with locally drifting cell boundaries.

    Simulates AI-generated pixel art where the grid isn't perfectly uniform.
    Returns the image plus the actual boundary positions used.
    """
    rng = np.random.RandomState(seed)
    palette = rng.randint(30, 226, size=(num_colors, 3), dtype=np.uint8)
    color_indices = rng.randint(0, num_colors, size=(grid_height, grid_width))

    # Generate drifted boundaries
    x_boundaries = [0]
    for i in range(1, grid_width):
        ideal = round(i * grid_width * cell_size / grid_width)
        drift = rng.randint(-max_drift, max_drift + 1)
        x_boundaries.append(max(x_boundaries[-1] + 2, ideal + drift))
    x_boundaries.append(grid_width * cell_size)

    y_boundaries = [0]
    for i in range(1, grid_height):
        ideal = round(i * grid_height * cell_size / grid_height)
        drift = rng.randint(-max_drift, max_drift + 1)
        y_boundaries.append(max(y_boundaries[-1] + 2, ideal + drift))
    y_boundaries.append(grid_height * cell_size)

    img_h = grid_height * cell_size
    img_w = grid_width * cell_size
    pixels = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    for row in range(grid_height):
        for col in range(grid_width):
            color = palette[color_indices[row, col]]
            y0 = y_boundaries[row]
            y1 = y_boundaries[row + 1]
            x0 = x_boundaries[col]
            x1 = x_boundaries[col + 1]
            pixels[y0:y1, x0:x1] = color

    img = Image.fromarray(pixels, "RGB")
    return img, x_boundaries, y_boundaries


@pytest.fixture
def drifted_grid():
    """16x16 grid with ±2px boundary drift in a 256x256 image."""
    img, x_bounds, y_bounds = make_drifted_pixel_art(16, 16, 16)
    return img, 16, 16, 16, x_bounds, y_bounds


@pytest.fixture
def small_grid():
    """8x8 grid in a 128x128 image (cell_size=16)."""
    img, palette, indices = make_pixel_art(8, 8, 16)
    return img, 8, 8, 16, palette, indices


@pytest.fixture
def medium_grid():
    """32x32 grid in a 512x512 image (cell_size=16)."""
    img, palette, indices = make_pixel_art(32, 32, 16)
    return img, 32, 32, 16, palette, indices


@pytest.fixture
def non_power_of_2_grid_24():
    """24x24 grid in a 480x480 image (cell_size=20)."""
    img, palette, indices = make_pixel_art(24, 24, 20)
    return img, 24, 24, 20, palette, indices


@pytest.fixture
def non_power_of_2_grid_20():
    """20x20 grid in a 400x400 image (cell_size=20)."""
    img, palette, indices = make_pixel_art(20, 20, 20)
    return img, 20, 20, 20, palette, indices


@pytest.fixture
def antialiased_grid():
    """32x32 grid with anti-aliasing in a 512x512 image."""
    img, palette, indices = make_pixel_art(32, 32, 16, antialiased=True)
    return img, 32, 32, 16, palette, indices


@pytest.fixture
def rgba_grid():
    """16x16 RGBA grid in a 256x256 image (cell_size=16)."""
    img, palette, indices = make_pixel_art(16, 16, 16, rgba=True)
    return img, 16, 16, 16, palette, indices
