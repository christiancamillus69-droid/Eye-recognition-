"""Synthetic boards that behave like photographs.

`render.py` draws flat glyphs on flat colour and warps the result. That is
enough to test geometry, but it quietly cheats in every way that matters for a
real photo, so passing on it says less than it appears to.

This module removes the cheats one at a time:

*Pieces stand up.* They are drawn in scene space, anchored at their square and
extending upward, back rank first so nearer pieces occlude farther ones. In the
rectified image a piece therefore smears away from the camera and overhangs the
square behind it — which is the entire reason
:data:`boardeye.squares.HEADROOM_RATIO` exists, and which flat glyphs never
exercised.

*Pieces cast shadows.* A shadow falls across neighbouring squares and is the
most plausible way to fool occupancy into seeing a piece that is not there.

*The board has grain and the light is uneven.* Wood texture adds edge noise
that competes with the grid lines during detection, and a lighting gradient
plus vignette breaks any assumption that one square's brightness predicts
another's.

*The table is cluttered.* A board on a plain backdrop makes contour detection
far easier than it will ever be in a kitchen.

None of this makes it a photograph. It does mean a pass here is evidence rather
than decoration.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from boardeye.fen import placement_to_grid

from .render import FONT_PATH, SOLID_GLYPH

#: Height of each piece as a fraction of one square's width.
PIECE_HEIGHT = {"p": 0.95, "n": 1.15, "b": 1.25, "r": 1.05, "q": 1.40, "k": 1.55}


@dataclass
class Realism:
    """How unkind the simulated photograph should be."""

    tilt: float = 0.16          # perspective compression of the far edge
    roll: float = 2.0           # camera not held square, in degrees
    grain: float = 0.05         # wood texture strength
    light_gradient: float = 0.22  # brightness falloff across the board
    vignette: float = 0.25      # darkening toward the frame corners
    shadow: float = 0.45        # cast-shadow opacity
    clutter: bool = True        # objects on the table around the board
    blur: float = 0.7
    noise: float = 0.012
    jpeg_quality: int = 82

    @classmethod
    def gentle(cls) -> "Realism":
        return cls(tilt=0.08, roll=1.0, grain=0.03, light_gradient=0.12,
                   vignette=0.15, shadow=0.30, blur=0.4, noise=0.006, jpeg_quality=92)

    @classmethod
    def typical(cls) -> "Realism":
        return cls()

    @classmethod
    def harsh(cls) -> "Realism":
        return cls(tilt=0.28, roll=4.5, grain=0.09, light_gradient=0.38,
                   vignette=0.38, shadow=0.65, blur=1.3, noise=0.03, jpeg_quality=62)


LIGHT_SQUARE = np.array([181, 217, 240], dtype=np.float32)  # BGR
DARK_SQUARE = np.array([99, 136, 181], dtype=np.float32)
WHITE_PIECE = (240, 246, 248)
BLACK_PIECE = (28, 30, 32)


def _wood_board(square_px: int, grain: float, rng) -> np.ndarray:
    """An 8x8 board with wood grain and visible square boundaries."""
    size = square_px * 8
    board = np.zeros((size, size, 3), dtype=np.float32)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    # Long streaks along one axis, the way sawn timber looks.
    streaks = (
        np.sin(yy * 0.35 + np.sin(xx * 0.02) * 4.0) * 0.5
        + np.sin(yy * 0.11 + 1.7) * 0.3
        + rng.normal(0, 0.4, (size, size))
    )
    streaks = cv2.GaussianBlur(streaks, (0, 0), 1.2)

    for r in range(8):
        for f in range(8):
            colour = DARK_SQUARE if (r + f) % 2 else LIGHT_SQUARE
            y0, y1 = r * square_px, (r + 1) * square_px
            x0, x1 = f * square_px, (f + 1) * square_px
            patch = np.broadcast_to(colour, (y1 - y0, x1 - x0, 3)).copy()
            board[y0:y1, x0:x1] = patch * (
                1.0 + grain * streaks[y0:y1, x0:x1][:, :, None]
            )

    # Thin darker lines between squares, as most real boards have.
    for i in range(9):
        p = min(size - 1, i * square_px)
        width = max(1, square_px // 40)
        board[max(0, p - width) : p + width, :] *= 0.82
        board[:, max(0, p - width) : p + width] *= 0.82

    return np.clip(board, 0, 255).astype(np.uint8)


def _homography(board_px: int, canvas, margin: int, tilt: float, roll: float):
    cw, ch = canvas
    inner_w, inner_h = cw - 2 * margin, ch - 2 * margin
    inset = tilt * inner_w
    dst = np.float32(
        [
            [margin + inset, margin],
            [margin + inner_w - inset, margin],
            [margin + inner_w, margin + inner_h],
            [margin, margin + inner_h],
        ]
    )
    if roll:
        centre = dst.mean(axis=0)
        angle = np.deg2rad(roll)
        rotation = np.float32(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        dst = (dst - centre) @ rotation.T + centre

    src = np.float32([[0, 0], [board_px, 0], [board_px, board_px], [0, board_px]])
    return cv2.getPerspectiveTransform(src, dst), dst


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homography to Nx2 points."""
    homogeneous = np.hstack([points, np.ones((len(points), 1), np.float32)])
    projected = (matrix @ homogeneous.T).T
    return projected[:, :2] / projected[:, 2:3]


def _clutter(scene: np.ndarray, board_quad: np.ndarray, rng) -> None:
    """Put plausible junk on the table so contour detection has competition."""
    h, w = scene.shape[:2]
    for _ in range(rng.integers(5, 10)):
        colour = tuple(int(c) for c in rng.integers(40, 200, 3))
        if rng.random() < 0.5:
            centre = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            axes = (int(rng.integers(20, 90)), int(rng.integers(15, 70)))
            cv2.ellipse(scene, centre, axes, float(rng.integers(0, 180)), 0, 360, colour, -1)
        else:
            # Rectangles are the dangerous kind: a book or a phone is exactly
            # the four-sided blob the board detector is looking for.
            x, y = int(rng.integers(0, w - 120)), int(rng.integers(0, h - 120))
            size = (int(rng.integers(60, 160)), int(rng.integers(50, 130)))
            box = cv2.boxPoints(
                ((x, y), size, float(rng.integers(0, 90)))
            ).astype(np.int32)
            cv2.fillConvexPoly(scene, box, colour)
    # Never let clutter cover the board itself.
    cv2.fillConvexPoly(scene, board_quad.astype(np.int32), (0, 0, 0))


def _lighting(scene: np.ndarray, gradient: float, vignette: float, rng) -> np.ndarray:
    h, w = scene.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    angle = rng.uniform(0, 2 * np.pi)
    ramp = (np.cos(angle) * (xx / w - 0.5) + np.sin(angle) * (yy / h - 0.5)) * 2.0
    light = 1.0 + gradient * ramp

    cx, cy = w / 2, h / 2
    radius = np.hypot(xx - cx, yy - cy) / np.hypot(cx, cy)
    light *= 1.0 - vignette * radius**2

    return np.clip(scene.astype(np.float32) * light[:, :, None], 0, 255).astype(np.uint8)


def render_photo(
    position: str | list[list[str]],
    realism: Realism | None = None,
    *,
    square_px: int = 96,
    canvas: tuple[int, int] = (1000, 1000),
    margin: int = 90,
    from_black_side: bool = False,
    seed: int | None = None,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Render a position as something resembling a photograph of a real set.

    Set ``from_black_side`` to photograph the board from the other end, which
    reverses the direction pieces lean in the rectified image.

    Returns the image and the board's four corners within it, ordered
    top-left, top-right, bottom-right, bottom-left.
    """
    realism = realism or Realism.typical()
    rng = np.random.default_rng(seed)
    grid = (
        placement_to_grid(position.split()[0]) if isinstance(position, str) else position
    )
    if from_black_side:
        grid = [row[::-1] for row in grid[::-1]]

    board_px = square_px * 8
    board = _wood_board(square_px, realism.grain, rng)

    matrix, quad = _homography(board_px, canvas, margin, realism.tilt, realism.roll)
    cw, ch = canvas

    # Table first, then the board warped onto it.
    scene = np.full((ch, cw, 3), (78, 88, 104), dtype=np.uint8)
    scene = _lighting(scene, realism.light_gradient * 0.5, 0.0, rng)
    if realism.clutter:
        _clutter(scene, quad, rng)

    warped_board = cv2.warpPerspective(board, matrix, (cw, ch))
    mask = cv2.warpPerspective(
        np.full((board_px, board_px), 255, np.uint8), matrix, (cw, ch),
        flags=cv2.INTER_NEAREST,
    )
    scene[mask > 0] = warped_board[mask > 0]

    _draw_pieces(scene, grid, matrix, square_px, realism, rng)

    scene = _lighting(scene, realism.light_gradient, realism.vignette, rng)

    if realism.blur:
        scene = np.array(Image.fromarray(scene).filter(ImageFilter.GaussianBlur(realism.blur)))
    if realism.noise:
        scene = np.clip(
            scene.astype(np.float32) + rng.normal(0, realism.noise * 255, scene.shape),
            0, 255,
        ).astype(np.uint8)
    if realism.jpeg_quality < 100:
        ok, buffer = cv2.imencode(
            ".jpg", scene, [int(cv2.IMWRITE_JPEG_QUALITY), realism.jpeg_quality]
        )
        if ok:
            scene = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    return scene, [(float(x), float(y)) for x, y in quad]


def _draw_pieces(scene, grid, matrix, square_px, realism: Realism, rng) -> None:
    """Draw standing pieces with cast shadows, farthest rank first.

    Painting back to front is what produces occlusion: a tall piece on rank 4
    partly hides the one behind it on rank 5, exactly as in a real photograph
    and exactly the thing that makes a far-rank square hard to read.
    """
    overlay = Image.fromarray(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(overlay, "RGBA")

    # Shadows all go on before any piece, so a piece is never shadowed by a
    # neighbour drawn after it.
    for r in range(8):
        for f in range(8):
            if grid[r][f]:
                _draw_shadow(draw, grid[r][f], r, f, matrix, square_px, realism)

    for r in range(8):  # rank 8 first == farthest from the camera
        for f in range(8):
            if grid[r][f]:
                _draw_piece(draw, grid[r][f], r, f, matrix, square_px)

    scene[:] = cv2.cvtColor(np.array(overlay.convert("RGB")), cv2.COLOR_RGB2BGR)


def _square_geometry(r: int, f: int, matrix, square_px: int):
    """Where a square's centre lands, and how big a square is there."""
    centre = _project(
        matrix, np.float32([[(f + 0.5) * square_px, (r + 0.9) * square_px]])
    )[0]
    edge = _project(
        matrix,
        np.float32(
            [[(f + 0.5) * square_px, (r + 0.9) * square_px - square_px]]
        ),
    )[0]
    scale = float(np.hypot(*(edge - centre)))
    return centre, scale


def _draw_shadow(draw, piece, r, f, matrix, square_px, realism: Realism) -> None:
    centre, scale = _square_geometry(r, f, matrix, square_px)
    height = PIECE_HEIGHT[piece.lower()]
    length = scale * height * 0.7
    # Shadow lies on the board, thrown to one side — it will cross onto
    # neighbouring squares, which is precisely the risk to occupancy.
    tip = (centre[0] + length * 0.75, centre[1] + length * 0.25)
    width = scale * 0.34
    draw.polygon(
        [
            (centre[0] - width, centre[1]),
            (centre[0] + width, centre[1]),
            (tip[0] + width * 0.6, tip[1]),
            (tip[0] - width * 0.6, tip[1]),
        ],
        fill=(0, 0, 0, int(255 * realism.shadow * 0.5)),
    )


def _draw_piece(draw, piece, r, f, matrix, square_px: int) -> None:
    centre, scale = _square_geometry(r, f, matrix, square_px)
    height = PIECE_HEIGHT[piece.lower()]
    glyph_px = max(8, int(round(scale * height)))

    font = ImageFont.truetype(FONT_PATH, glyph_px)
    is_white = piece.isupper()
    fill = WHITE_PIECE if is_white else BLACK_PIECE
    stroke = BLACK_PIECE if is_white else WHITE_PIECE

    # Anchored at the base ("ms" = middle of the baseline), so the piece stands
    # on its square and extends upward into the square behind it.
    draw.text(
        (centre[0], centre[1]),
        SOLID_GLYPH[piece.lower()],
        font=font,
        fill=tuple(int(c) for c in fill),
        anchor="ms",
        stroke_width=max(1, glyph_px // 26),
        stroke_fill=tuple(int(c) for c in stroke),
    )


STARTING_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
