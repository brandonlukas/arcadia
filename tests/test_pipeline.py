"""End-to-end pipeline tests."""

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from arcadia.grid import GridDetectionError
from arcadia.pipeline import convert
from tests.conftest import make_pixel_art


class TestConvert:
    def test_basic_e2e(self, tmp_path, small_grid):
        img, gw, gh, cs, _, _ = small_grid
        output = tmp_path / "out.svg"
        convert(img, output)
        assert output.exists()
        svg = output.read_text()
        root = ET.fromstring(svg)
        assert root.attrib["viewBox"] == f"0 0 {gw} {gh}"

    def test_e2e_32x32(self, tmp_path, medium_grid):
        img, gw, gh, cs, _, _ = medium_grid
        output = tmp_path / "out.svg"
        convert(img, output)
        svg = output.read_text()
        root = ET.fromstring(svg)
        ns = {"svg": "http://www.w3.org/2000/svg"}
        rects = root.findall("svg:rect", ns)
        assert len(rects) == gw * gh

    def test_e2e_antialiased(self, tmp_path, antialiased_grid):
        img, gw, gh, cs, _, _ = antialiased_grid
        output = tmp_path / "out.svg"
        convert(img, output)
        svg = output.read_text()
        root = ET.fromstring(svg)
        ns = {"svg": "http://www.w3.org/2000/svg"}
        rects = root.findall("svg:rect", ns)
        assert len(rects) == gw * gh

    def test_e2e_rgba(self, tmp_path, rgba_grid):
        img, gw, gh, cs, _, _ = rgba_grid
        # Convert RGBA to RGB (as CLI does)
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        output = tmp_path / "out.svg"
        convert(background, output)
        assert output.exists()

    def test_e2e_non_power_of_2(self, tmp_path, non_power_of_2_grid_24):
        img, gw, gh, cs, _, _ = non_power_of_2_grid_24
        output = tmp_path / "out.svg"
        convert(img, output)
        svg = output.read_text()
        root = ET.fromstring(svg)
        assert root.attrib["viewBox"] == f"0 0 {gw} {gh}"

    def test_photograph_raises(self, tmp_path):
        rng = np.random.RandomState(99)
        noise = rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(noise, "RGB")
        output = tmp_path / "out.svg"
        with pytest.raises(GridDetectionError):
            convert(img, output)

    def test_verbose_output(self, tmp_path, small_grid, capsys):
        img, gw, gh, cs, _, _ = small_grid
        output = tmp_path / "out.svg"
        convert(img, output, verbose=True)
        captured = capsys.readouterr()
        assert "Grid detected" in captured.out
        assert "Palette" in captured.out
        assert "SVG rendered" in captured.out

    def test_performance_512x512(self, tmp_path):
        """512x512 image should convert in well under 10 seconds."""
        img, _, _ = make_pixel_art(32, 32, 16)
        output = tmp_path / "out.svg"
        t0 = time.perf_counter()
        convert(img, output)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"Conversion took {elapsed:.2f}s, expected < 10s"
