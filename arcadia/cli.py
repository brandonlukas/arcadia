"""CLI entry point for Arcadia."""

import argparse
import sys
from pathlib import Path

from PIL import Image

from arcadia.grid import GridDetectionError
from arcadia.pipeline import convert


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="arcadia",
        description="Convert AI-generated pixel art PNGs into true pixel art SVGs.",
    )
    parser.add_argument("input", type=Path, help="Path to input PNG file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Path to output SVG file (default: input with .svg extension)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print diagnostic information during conversion",
    )
    return parser.parse_args(argv)


def load_image(path: Path) -> Image.Image:
    """Load a PNG and convert to RGB (flatten RGBA with white matte).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be opened as an image.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        img = Image.open(path)
    except Exception as e:
        raise ValueError(f"Could not open image: {e}") from e
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        return background
    return img.convert("RGB")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output or args.input.with_suffix(".svg")

    try:
        img = load_image(args.input)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Loaded: {args.input} ({img.width}x{img.height})")

    try:
        convert(img, output, verbose=args.verbose)
    except GridDetectionError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Output: {output}")


if __name__ == "__main__":
    main()
