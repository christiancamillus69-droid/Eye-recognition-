"""Synthetic board images with known ground truth.

The point of this module is to let the CV and classifier tests assert against
positions we already know the answer to, without needing a camera or a corpus
of photographs. It renders a board from a FEN, then optionally simulates
photographing it: perspective tilt, a surrounding background, blur and noise.

This is a test fixture, not part of the shipped package. It stands in for real
photographs well enough to catch geometry and plumbing bugs, but it cannot tell
us how the classifier will do on real wooden pieces — only a real photo can,
and that test is listed in the README as one for the user to run.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from boardeye.fen import placement_to_grid

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# The solid (nominally "black") glyphs are used for both colours and recoloured,
# because that is what real pieces look like: solid objects in a light or dark
# finish, not hollow outlines.
SOLID_GLYPH = {
    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}

LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
WHITE_PIECE = (248, 246, 240)
BLACK_PIECE = (32, 30, 28)
GRID_LINE = (120, 90, 62)


def _as_grid(position: str | list[list[str]]) -> list[list[str]]:
    if isinstance(position, str):
        return placement_to_grid(position.split()[0])
    return position


def render_board(
    position: str | list[list[str]],
    square_px: int = 64,
    *,
    draw_grid_lines: bool = True,
    piece_scale: float = 0.80,
    light: tuple[int, int, int] = LIGHT_SQUARE,
    dark: tuple[int, int, int] = DARK_SQUARE,
) -> np.ndarray:
    """Render a top-down board. Returns a BGR array, board filling the frame.

    Grid convention matches :mod:`boardeye.types`: row 0 is rank 8.
    """
    grid = _as_grid(position)
    size = square_px * 8
    img = Image.new("RGB", (size, size), light)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, int(square_px * piece_scale))

    for r in range(8):
        for f in range(8):
            x0, y0 = f * square_px, r * square_px
            box = (x0, y0, x0 + square_px, y0 + square_px)
            # a1 is dark; a1 sits at row 7, file 0, so parity is (r + f) % 2.
            if (r + f) % 2 == 1:
                draw.rectangle(box, fill=dark)
            if draw_grid_lines:
                draw.rectangle(box, outline=GRID_LINE, width=max(1, square_px // 64))

    for r in range(8):
        for f in range(8):
            piece = grid[r][f]
            if not piece:
                continue
            glyph = SOLID_GLYPH[piece.lower()]
            is_white = piece.isupper()
            fill = WHITE_PIECE if is_white else BLACK_PIECE
            stroke = BLACK_PIECE if is_white else WHITE_PIECE
            cx = f * square_px + square_px / 2
            cy = r * square_px + square_px / 2
            draw.text(
                (cx, cy),
                glyph,
                font=font,
                fill=fill,
                anchor="mm",
                stroke_width=max(1, square_px // 40),
                stroke_fill=stroke,
            )

    return np.array(img)[:, :, ::-1].copy()  # RGB -> BGR for OpenCV


def simulate_photo(
    board: np.ndarray,
    *,
    canvas: tuple[int, int] = (900, 900),
    tilt: float = 0.0,
    margin: int = 60,
    background: tuple[int, int, int] = (70, 78, 86),
    blur: float = 0.0,
    noise: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Place a rendered board into a scene, optionally tilted.

    ``tilt`` is a fraction of the board width by which the top edge is pulled
    inward, simulating a photo taken from in front of the board rather than
    directly above. 0.0 is perfectly top-down; 0.25 is a steep angle.

    Returns the photo and the four board corners within it, ordered
    top-left, top-right, bottom-right, bottom-left — the same order
    :func:`boardeye.detect.detect_board` reports.
    """
    import cv2

    rng = np.random.default_rng(seed)
    h, w = board.shape[:2]
    cw, ch = canvas

    inner_w = cw - 2 * margin
    inner_h = ch - 2 * margin
    inset = tilt * inner_w

    dst = np.float32(
        [
            [margin + inset, margin],
            [margin + inner_w - inset, margin],
            [margin + inner_w, margin + inner_h],
            [margin, margin + inner_h],
        ]
    )
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    matrix = cv2.getPerspectiveTransform(src, dst)
    scene = np.full((ch, cw, 3), background, dtype=np.uint8)
    warped = cv2.warpPerspective(board, matrix, (cw, ch), borderMode=cv2.BORDER_TRANSPARENT)

    # Composite only where the board actually landed, so the background shows
    # through everywhere else.
    mask = cv2.warpPerspective(
        np.full((h, w), 255, np.uint8), matrix, (cw, ch), flags=cv2.INTER_NEAREST
    )
    scene[mask > 0] = warped[mask > 0]

    if blur > 0:
        scene = np.array(Image.fromarray(scene).filter(ImageFilter.GaussianBlur(blur)))
    if noise > 0:
        noisy = scene.astype(np.float32) + rng.normal(0, noise * 255, scene.shape)
        scene = np.clip(noisy, 0, 255).astype(np.uint8)

    corners = [(float(x), float(y)) for x, y in dst]
    return scene, corners


def render_photo(
    position: str | list[list[str]],
    *,
    square_px: int = 64,
    **photo_kwargs,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Convenience: render a position and immediately simulate photographing it."""
    return simulate_photo(render_board(position, square_px=square_px), **photo_kwargs)


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
