"""Cut a rectified board into squares and read what is geometrically knowable.

Two questions get answered here — is this square occupied, and is the piece on
it light or dark — and both are answered without any learned model. They are
photometric questions, which is why they are the dependable part of the
pipeline; piece *type* is the hard part and lives in :mod:`boardeye.classifier`.

Two ideas keep this robust across boards and lighting:

*Each square is its own reference.* Background colour is measured from a ring
around the square's own border, which stays visible even when a piece sits in
the middle. Occupancy is then deviation from that local reference, so a shadow
falling across one corner of the board does not shift the whole board's
threshold.

*Piece colour is decided by clustering, not by a fixed threshold.* We collect
every occupied square's piece brightness and split the population in two. A
"white" piece under warm indoor light may be darker in absolute terms than a
"black" piece under a window, so only the relative split is meaningful.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detect import BOARD_PX
from .types import BoardReading, SquareReading

#: Fraction of the square used as the background reference ring.
RING_RATIO = 0.18
#: Fraction of a square's height added above a crop, so tall pieces that lean
#: toward the camera are not decapitated. Only used for classifier crops.
HEADROOM_RATIO = 0.55

#: A square is called occupied above this fraction of deviating interior pixels.
OCCUPANCY_THRESHOLD = 0.16
#: Colour distance (0-255 scale) beyond which a pixel counts as "not background".
BACKGROUND_TOLERANCE = 38.0
#: Minimum separation between the two brightness clusters for the light/dark
#: split to be believable. Below this we are probably looking at pieces of only
#: one colour and say so by dropping confidence.
MIN_COLOUR_SEPARATION = 25.0


def _confidence(value: float, threshold: float, scale: float) -> float:
    """Map a score to [0, 1]: 0 right at the decision boundary, 1 far from it."""
    z = (value - threshold) / max(scale, 1e-6)
    probability = 1.0 / (1.0 + np.exp(-z))
    return float(min(1.0, 2.0 * abs(probability - 0.5)))


def square_bounds(index: int, board_px: int = BOARD_PX) -> tuple[int, int]:
    """Pixel span of one rank or file within the rectified board."""
    size = board_px / 8.0
    return int(round(index * size)), int(round((index + 1) * size))


def cell(warped: np.ndarray, rank: int, file: int) -> np.ndarray:
    """The square's own pixels, with no headroom. Used for occupancy/colour."""
    board_px = warped.shape[0]
    y0, y1 = square_bounds(rank, board_px)
    x0, x1 = square_bounds(file, board_px)
    return warped[y0:y1, x0:x1]


def piece_crop(
    warped: np.ndarray, rank: int, file: int, headroom: float = HEADROOM_RATIO
) -> np.ndarray:
    """The square plus headroom above it, which is what the classifier sees.

    Pieces are three-dimensional: in anything but a perfectly overhead shot the
    top of a piece appears above its own square. Cropping to the square alone
    would cut the crown off a king and leave it looking like a pawn.

    Squares on rank 8 have no board above them to borrow headroom from, so the
    missing rows are padded by replicating the top edge. Padding rather than
    clipping keeps every crop the same shape, which matters because
    :func:`boardeye.classifier.piece_map` locates the headroom band by
    proportion — a short crop would have it slice into the piece instead, and
    the back-rank pieces are exactly the ones calibration can least afford to
    corrupt.
    """
    board_px = warped.shape[0]
    y0, y1 = square_bounds(rank, board_px)
    x0, x1 = square_bounds(file, board_px)
    extra = int(round((y1 - y0) * headroom))

    top = max(0, y0 - extra)
    crop = warped[top:y1, x0:x1]
    missing = extra - (y0 - top)
    if missing > 0:
        crop = cv2.copyMakeBorder(crop, missing, 0, 0, 0, cv2.BORDER_REPLICATE)
    return crop


def _ring_and_interior(square: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a square into its border ring and its interior."""
    h, w = square.shape[:2]
    band_y = max(1, int(round(h * RING_RATIO)))
    band_x = max(1, int(round(w * RING_RATIO)))
    mask = np.zeros((h, w), dtype=bool)
    mask[:band_y, :] = True
    mask[-band_y:, :] = True
    mask[:, :band_x] = True
    mask[:, -band_x:] = True
    return square[mask], square[~mask]


def _occupancy_score(square: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Fraction of interior pixels unlike the square's own background.

    Returns the score, the interior pixels, and the boolean mask selecting the
    ones judged to belong to a piece.
    """
    ring, interior = _ring_and_interior(square)
    if ring.size == 0 or interior.size == 0:
        return 0.0, interior, np.zeros(0, dtype=bool)

    background = np.median(ring.reshape(-1, ring.shape[-1]), axis=0)
    flat = interior.reshape(-1, interior.shape[-1]).astype(np.float32)
    distance = np.linalg.norm(flat - background[None, :], axis=1)
    piece_mask = distance > BACKGROUND_TOLERANCE
    return float(piece_mask.mean()), flat, piece_mask


def _luminance(pixels: np.ndarray) -> float:
    """Perceptual brightness of BGR pixels (Rec. 601 weights)."""
    if pixels.size == 0:
        return float("nan")
    b, g, r = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    return float(np.median(0.114 * b + 0.587 * g + 0.299 * r))


def analyse(warped: np.ndarray) -> BoardReading:
    """Read occupancy and piece colour for all 64 squares.

    ``piece`` is left empty on every square: naming the piece is the
    classifier's job. This function fills in everything that can be known
    without a trained model, which is also everything the classifier needs in
    order to be asked only about squares that actually hold a piece.
    """
    squares = [[SquareReading() for _ in range(8)] for _ in range(8)]

    brightnesses: list[tuple[int, int, float]] = []
    for r in range(8):
        for f in range(8):
            score, flat, piece_mask = _occupancy_score(cell(warped, r, f))
            reading = squares[r][f]
            reading.occupied = score >= OCCUPANCY_THRESHOLD
            reading.occupancy_confidence = _confidence(
                score, OCCUPANCY_THRESHOLD, scale=0.06
            )
            if reading.occupied and piece_mask.any():
                brightnesses.append((r, f, _luminance(flat[piece_mask])))

    _assign_colours(squares, brightnesses)
    return BoardReading(squares=squares, warped=warped)


def _assign_colours(
    squares: list[list[SquareReading]], brightnesses: list[tuple[int, int, float]]
) -> None:
    """Split occupied squares into light-pieced and dark-pieced by clustering."""
    if not brightnesses:
        return

    values = np.array([b for _, _, b in brightnesses], dtype=np.float32)
    if len(values) == 1:
        # Nothing to compare against; call it and admit we are guessing.
        r, f, _ = brightnesses[0]
        squares[r][f].is_black = bool(values[0] < 128)
        squares[r][f].colour_confidence = 0.0
        return

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centres = cv2.kmeans(
        values.reshape(-1, 1), 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.ravel()
    centres = centres.ravel()

    dark_label = int(np.argmin(centres))
    separation = float(abs(centres[0] - centres[1]))
    boundary = float(np.mean(centres))

    # When the two clusters nearly coincide, the pieces on the board are
    # probably all one colour and the split is meaningless. Keep the call but
    # report no confidence, so the editor surfaces these squares for review.
    believable = separation >= MIN_COLOUR_SEPARATION

    for (r, f, brightness), label in zip(brightnesses, labels):
        reading = squares[r][f]
        reading.is_black = bool(label == dark_label)
        reading.colour_confidence = (
            _confidence(brightness, boundary, scale=separation / 4.0)
            if believable
            else 0.0
        )


def crops_for_occupied(
    reading: BoardReading, headroom: float = HEADROOM_RATIO
) -> list[tuple[int, int, np.ndarray]]:
    """Classifier-ready crops for every square judged occupied."""
    if reading.warped is None:
        raise ValueError("BoardReading has no warped image to crop from")
    out = []
    for r in range(8):
        for f in range(8):
            if reading.squares[r][f].occupied:
                out.append((r, f, piece_crop(reading.warped, r, f, headroom)))
    return out
