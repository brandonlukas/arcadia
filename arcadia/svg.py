"""SVG generation from quantized pixel art grid."""

from arcadia.alignment import AlignedGrid
from arcadia.grid import GridResult
from arcadia.palette import PaletteResult


def render(palette_result: PaletteResult, grid: AlignedGrid | GridResult) -> str:
    """Render a pixel art SVG from the quantized color grid.

    Each logical pixel becomes one <rect> element.
    The viewBox is set to grid dimensions so each pixel = 1 SVG unit.

    Args:
        palette_result: Quantized color grid and palette.
        grid: Detected grid parameters.

    Returns:
        SVG string.
    """
    lines: list[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {grid.grid_width} {grid.grid_height}" '
        f'shape-rendering="crispEdges">'
    )

    for row_idx, color_row in enumerate(palette_result.color_grid):
        for col_idx, color in enumerate(color_row):
            lines.append(
                f'  <rect x="{col_idx}" y="{row_idx}" '
                f'width="1" height="1" fill="{color}"/>'
            )

    lines.append("</svg>")
    return "\n".join(lines)
