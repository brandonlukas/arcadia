"""Tests for SVG generation module."""

import xml.etree.ElementTree as ET

from arcadia.grid import GridResult
from arcadia.palette import PaletteResult
from arcadia.svg import render


def _make_simple_palette(width: int, height: int) -> PaletteResult:
    """Create a simple checkered pattern palette result."""
    colors = ["#ff0000", "#0000ff"]
    grid = []
    for row in range(height):
        row_colors = []
        for col in range(width):
            row_colors.append(colors[(row + col) % 2])
        grid.append(row_colors)
    return PaletteResult(color_grid=grid, palette=colors)


class TestRender:
    def test_valid_svg(self):
        grid = GridResult(cell_size=16, grid_width=4, grid_height=4, confidence=0.9)
        palette = _make_simple_palette(4, 4)
        svg = render(palette, grid)
        # Should be parseable XML
        root = ET.fromstring(svg)
        assert root.tag == "{http://www.w3.org/2000/svg}svg"

    def test_viewbox(self):
        grid = GridResult(cell_size=16, grid_width=32, grid_height=32, confidence=0.9)
        palette = _make_simple_palette(32, 32)
        svg = render(palette, grid)
        root = ET.fromstring(svg)
        assert root.attrib["viewBox"] == "0 0 32 32"

    def test_crisp_edges(self):
        grid = GridResult(cell_size=16, grid_width=4, grid_height=4, confidence=0.9)
        palette = _make_simple_palette(4, 4)
        svg = render(palette, grid)
        assert 'shape-rendering="crispEdges"' in svg

    def test_rect_count(self):
        grid = GridResult(cell_size=16, grid_width=8, grid_height=8, confidence=0.9)
        palette = _make_simple_palette(8, 8)
        svg = render(palette, grid)
        root = ET.fromstring(svg)
        ns = {"svg": "http://www.w3.org/2000/svg"}
        rects = root.findall("svg:rect", ns)
        assert len(rects) == 64  # 8x8

    def test_rect_colors(self):
        grid = GridResult(cell_size=16, grid_width=2, grid_height=2, confidence=0.9)
        palette = PaletteResult(
            color_grid=[["#ff0000", "#00ff00"], ["#0000ff", "#ffff00"]],
            palette=["#0000ff", "#00ff00", "#ff0000", "#ffff00"],
        )
        svg = render(palette, grid)
        root = ET.fromstring(svg)
        ns = {"svg": "http://www.w3.org/2000/svg"}
        rects = root.findall("svg:rect", ns)
        fills = [r.attrib["fill"] for r in rects]
        assert fills == ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]

    def test_rect_positions(self):
        grid = GridResult(cell_size=16, grid_width=3, grid_height=2, confidence=0.9)
        palette = _make_simple_palette(3, 2)
        svg = render(palette, grid)
        root = ET.fromstring(svg)
        ns = {"svg": "http://www.w3.org/2000/svg"}
        rects = root.findall("svg:rect", ns)
        # First row: (0,0), (1,0), (2,0)
        assert rects[0].attrib["x"] == "0"
        assert rects[0].attrib["y"] == "0"
        assert rects[2].attrib["x"] == "2"
        assert rects[2].attrib["y"] == "0"
        # Second row: (0,1), (1,1), (2,1)
        assert rects[3].attrib["x"] == "0"
        assert rects[3].attrib["y"] == "1"
