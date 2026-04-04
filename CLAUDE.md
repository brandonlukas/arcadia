# Arcadia

Pixel art PNG to SVG converter. Converts AI-generated "pixel art" images into true pixel art SVGs with one `<rect>` per logical pixel.

## Usage

```bash
arcadia input.png                    # → input.svg
arcadia input.png -o output.svg      # custom output path
arcadia input.png -v                 # verbose (grid size, palette count, timing)
```

## Development

- **Python:** >= 3.10
- **Python environment:** Always use conda environment `arcadia` — never install into base
  ```bash
  conda activate arcadia
  # or prefix commands with:
  conda run -n arcadia <command>
  ```
- **Install:** `conda run -n arcadia pip install -e .`
- **Tests:** `conda run -n arcadia python -m pytest tests/ -v`
- **Single test:** `conda run -n arcadia python -m pytest tests/test_grid.py::TestDetectGrid::test_8x8_grid -v`
- **Dependencies:** Pillow, numpy (no scipy)

## Architecture

```
arcadia/
  cli.py       — CLI entry point (argparse), image loading, RGBA→RGB flattening
  grid.py      — Grid detection via gradient autocorrelation
  alignment.py — Local grid alignment (snaps boundaries to actual edges)
  palette.py   — Per-cell color extraction + centroid-based palette merging
  svg.py       — SVG string generation (one rect per pixel, crispEdges)
  pipeline.py  — Orchestrates grid → palette → svg
```

Data flow: `PNG → load+RGB convert → detect_grid() → align_grid() → quantize() → render() → SVG file`

## Key Types

- `GridResult(cell_size, grid_width, grid_height, confidence)` — returned by `grid.detect_grid()`
- `AlignedGrid(x_boundaries, y_boundaries, grid_width, grid_height, cell_size, confidence)` — returned by `alignment.align_grid()`
- `PaletteResult(color_grid, palette)` — returned by `palette.quantize()`
- `GridDetectionError` — raised when input is not detectable pixel art (uniform image, photograph, low confidence)

## Key Design Decisions

- Grid detection uses **gradient autocorrelation**, not raw pixel autocorrelation — cell colors are random, but transitions between cells are periodic
- Color merging uses **Euclidean RGB distance** with threshold=30 and centroid comparison (not first-in-cluster)
- **Local grid alignment** snaps cell boundaries to actual color transition edges (±2px tolerance), handling AI images where the pixel grid drifts locally
- Grid phase detection (global offset from 0,0) is a future feature — local alignment partially compensates
- Animations, layering, and batch processing are deferred to future versions

## Gotchas

- `detect_grid()` works on the **gradient** (first-difference) of pixel rows, not raw pixels — raw autocorrelation fails because cell colors are random
- Images must be converted to RGB before passing to the pipeline — RGBA flattening uses white matte
- Color merge threshold (30 in RGB Euclidean space) may be too aggressive for subtle palettes — CIELAB Delta-E is a future improvement
- `align_grid()` uses full-image column/row gradient averages as a voting mechanism — robust even when some rows have no edges
