"""Pipeline orchestrating grid detection, alignment, color quantization, and SVG generation."""

import time
from pathlib import Path

from PIL import Image

from arcadia.alignment import align_grid
from arcadia.grid import detect_grid
from arcadia.palette import quantize
from arcadia.svg import render


def convert(image: Image.Image, output_path: Path, verbose: bool = False) -> None:
    """Convert an RGB PIL Image to a pixel art SVG.

    Args:
        image: RGB PIL Image (already loaded and converted from RGBA if needed).
        output_path: Where to write the output SVG file.
        verbose: Print diagnostic info.
    """
    t0 = time.perf_counter()

    grid = detect_grid(image)
    t_grid = time.perf_counter()
    if verbose:
        print(
            f"Grid detected: {grid.grid_width}x{grid.grid_height} "
            f"(cell_size={grid.cell_size}px, confidence={grid.confidence:.2f}) "
            f"[{t_grid - t0:.2f}s]"
        )

    aligned = align_grid(image, grid)
    t_align = time.perf_counter()
    if verbose:
        print(f"Grid aligned: local boundaries snapped [{t_align - t_grid:.2f}s]")

    palette_result = quantize(image, aligned)
    t_palette = time.perf_counter()
    if verbose:
        print(
            f"Palette: {len(palette_result.palette)} colors "
            f"[{t_palette - t_align:.2f}s]"
        )

    svg_str = render(palette_result, aligned)
    t_render = time.perf_counter()
    if verbose:
        print(f"SVG rendered: {len(svg_str)} bytes [{t_render - t_palette:.2f}s]")

    Path(output_path).write_text(svg_str, encoding="utf-8")

    t_total = time.perf_counter() - t0
    if verbose:
        print(f"Total: {t_total:.2f}s")
