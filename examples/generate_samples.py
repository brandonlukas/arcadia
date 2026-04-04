"""Generate sample pixel art PNGs for testing Arcadia.

Run: conda run -n arcadia python examples/generate_samples.py
"""

import numpy as np
from PIL import Image, ImageFilter

EXAMPLES_DIR = "examples"


def make_checkerboard(grid_size: int, cell_size: int, colors: list[tuple[int, int, int]]) -> np.ndarray:
    """Create a checkerboard pattern."""
    h = grid_size * cell_size
    w = grid_size * cell_size
    pixels = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(grid_size):
        for col in range(grid_size):
            color = colors[(row + col) % len(colors)]
            y0, y1 = row * cell_size, (row + 1) * cell_size
            x0, x1 = col * cell_size, (col + 1) * cell_size
            pixels[y0:y1, x0:x1] = color
    return pixels


def make_landscape(grid_w: int, grid_h: int, cell_size: int) -> np.ndarray:
    """Create a simple pixel art landscape (sky, hills, ground)."""
    h = grid_h * cell_size
    w = grid_w * cell_size
    pixels = np.zeros((h, w, 3), dtype=np.uint8)

    sky = (100, 149, 237)       # cornflower blue
    cloud = (255, 255, 255)     # white
    hill = (34, 139, 34)        # forest green
    ground = (139, 90, 43)      # brown
    grass = (50, 205, 50)       # lime green
    sun = (255, 223, 0)         # gold

    rng = np.random.RandomState(42)

    for row in range(grid_h):
        for col in range(grid_w):
            y0, y1 = row * cell_size, (row + 1) * cell_size
            x0, x1 = col * cell_size, (col + 1) * cell_size

            # Sky (top 60%)
            if row < grid_h * 0.6:
                color = sky
                # Sun
                if 2 <= row <= 4 and 2 <= col <= 4:
                    color = sun
                # Random clouds
                elif row < grid_h * 0.4 and rng.random() < 0.08:
                    color = cloud
            # Hills (60-75%)
            elif row < grid_h * 0.75:
                hill_height = int(3 * np.sin(col * 0.3) + 2 * np.sin(col * 0.7))
                if row < grid_h * 0.6 + max(0, hill_height):
                    color = sky
                else:
                    color = hill
            # Grass line
            elif row < grid_h * 0.8:
                color = grass
            # Ground
            else:
                color = ground

            pixels[y0:y1, x0:x1] = color

    return pixels


def make_character(cell_size: int) -> np.ndarray:
    """Create a simple 16x16 pixel art character sprite."""
    # 16x16 character design
    palette = {
        0: (100, 149, 237),  # background (sky blue)
        1: (50, 50, 50),     # outline (dark gray)
        2: (255, 200, 150),  # skin
        3: (200, 50, 50),    # shirt (red)
        4: (50, 50, 150),    # pants (blue)
        5: (139, 90, 43),    # hair (brown)
        6: (255, 255, 255),  # eyes (white)
    }

    # Simple character sprite
    grid = [
        [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
        [0,0,0,0,1,5,5,5,5,5,5,1,0,0,0,0],
        [0,0,0,1,5,5,5,5,5,5,5,5,1,0,0,0],
        [0,0,0,1,5,5,5,5,5,5,5,5,1,0,0,0],
        [0,0,0,1,2,2,2,2,2,2,2,2,1,0,0,0],
        [0,0,0,1,2,6,1,2,2,1,6,2,1,0,0,0],
        [0,0,0,1,2,2,2,1,1,2,2,2,1,0,0,0],
        [0,0,0,0,1,2,2,2,2,2,2,1,0,0,0,0],
        [0,0,0,1,3,3,3,3,3,3,3,3,1,0,0,0],
        [0,0,1,3,3,3,3,3,3,3,3,3,3,1,0,0],
        [0,0,1,3,3,3,3,3,3,3,3,3,3,1,0,0],
        [0,0,0,1,3,3,3,3,3,3,3,3,1,0,0,0],
        [0,0,0,1,4,4,4,4,4,4,4,4,1,0,0,0],
        [0,0,0,1,4,4,4,1,1,4,4,4,1,0,0,0],
        [0,0,0,1,4,4,4,1,1,4,4,4,1,0,0,0],
        [0,0,0,1,1,1,1,0,0,1,1,1,1,0,0,0],
    ]

    h = 16 * cell_size
    w = 16 * cell_size
    pixels = np.zeros((h, w, 3), dtype=np.uint8)

    for row in range(16):
        for col in range(16):
            color = palette[grid[row][col]]
            y0, y1 = row * cell_size, (row + 1) * cell_size
            x0, x1 = col * cell_size, (col + 1) * cell_size
            pixels[y0:y1, x0:x1] = color

    return pixels


def add_antialiasing(pixels: np.ndarray, radius: float = 0.8) -> np.ndarray:
    """Simulate AI-generated anti-aliasing by blurring slightly."""
    img = Image.fromarray(pixels, "RGB")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred)


def main():
    # 1. Clean 16x16 checkerboard (256x256, cell_size=16)
    checker = make_checkerboard(16, 16, [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0)])
    Image.fromarray(checker).save(f"{EXAMPLES_DIR}/checkerboard_16x16.png")
    print("Created checkerboard_16x16.png (256x256, 16x16 grid)")

    # 2. Landscape 32x24 (512x384, cell_size=16)
    landscape = make_landscape(32, 24, 16)
    Image.fromarray(landscape).save(f"{EXAMPLES_DIR}/landscape_32x24.png")
    print("Created landscape_32x24.png (512x384, 32x24 grid)")

    # 3. Landscape with AA (simulates AI-generated pixel art)
    landscape_aa = add_antialiasing(landscape, radius=1.0)
    Image.fromarray(landscape_aa).save(f"{EXAMPLES_DIR}/landscape_32x24_aa.png")
    print("Created landscape_32x24_aa.png (512x384, 32x24 grid, anti-aliased)")

    # 4. Character sprite 16x16 (256x256, cell_size=16)
    character = make_character(16)
    Image.fromarray(character).save(f"{EXAMPLES_DIR}/character_16x16.png")
    print("Created character_16x16.png (256x256, 16x16 grid)")

    # 5. Character sprite with AA
    character_aa = add_antialiasing(character, radius=1.2)
    Image.fromarray(character_aa).save(f"{EXAMPLES_DIR}/character_16x16_aa.png")
    print("Created character_16x16_aa.png (256x256, 16x16 grid, anti-aliased)")

    # 6. Large landscape 64x48 (1024x768, cell_size=16)
    large = make_landscape(64, 48, 16)
    large_aa = add_antialiasing(large, radius=1.0)
    Image.fromarray(large_aa).save(f"{EXAMPLES_DIR}/landscape_64x48_aa.png")
    print("Created landscape_64x48_aa.png (1024x768, 64x48 grid, anti-aliased)")

    print(f"\nDone! Try: arcadia examples/character_16x16_aa.png -v")


if __name__ == "__main__":
    main()
