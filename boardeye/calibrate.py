"""Teach BoardEye your pieces from a photo of the starting position.

This is the trick that makes an offline, dataset-free tool workable. Photograph
your board set up for a new game and the labels come free: the tool already
knows a rook stands on a1, so every occupied square hands it a labelled
training crop. One photo yields 32 samples spanning all six piece types, with
nothing to annotate by hand.

Two things are checked before anything is learned, because training on a
mislabelled photo would quietly poison the model:

*Is this actually the starting position?* The occupancy pattern must be two
full ranks at each end and four empty ranks between them. If it is not, we stop
and say so rather than learning that a knight is a bishop.

*Which way round is the board?* The photo could be taken from either side, or
in portrait. All four rotations are tried, the one matching the starting
occupancy pattern wins, and the light/dark split decides which end is white.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .classifier import PieceClassifier, TrainingStats
from .detect import BOARD_PX, detect_board, detect_from_corners
from .fen import placement_to_grid
from .squares import analyse, piece_crop
from .types import BoardReading

STARTING_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
STARTING_GRID = placement_to_grid(STARTING_PLACEMENT)
#: True where the starting position has a piece.
STARTING_OCCUPANCY = [[bool(cell) for cell in rank] for rank in STARTING_GRID]


class CalibrationError(Exception):
    """The photo could not be used to train."""


@dataclass
class CalibrationResult:
    samples_added: int
    rotation_applied: int
    stats: TrainingStats
    warnings: list[str] = field(default_factory=list)


def _rotate_squares(squares, quarter_turns: int):
    """Rotate a square grid clockwise, matching :func:`fen.rotate_grid`."""
    out = [row[:] for row in squares]
    for _ in range(quarter_turns % 4):
        out = [[out[7 - f][r] for f in range(8)] for r in range(8)]
    return out


def _occupancy_mismatch(squares) -> int:
    """How many squares disagree with the starting layout."""
    return sum(
        1
        for r in range(8)
        for f in range(8)
        if squares[r][f].occupied != STARTING_OCCUPANCY[r][f]
    )


def _best_rotation(reading: BoardReading) -> tuple[int, int]:
    """Rotation (in clockwise quarter turns) that best matches the start, and its error."""
    scored = [
        (_occupancy_mismatch(_rotate_squares(reading.squares, turns)), turns)
        for turns in range(4)
    ]
    scored.sort()
    return scored[0][1], scored[0][0]


def _white_is_at_bottom(squares) -> bool | None:
    """Decide which end of the board holds the light pieces.

    Occupancy alone cannot tell white from black — the starting position looks
    identical rotated 180 degrees. The colour clustering from square analysis
    can, so we count how many dark-clustered pieces sit at each end.
    """
    bottom = [squares[r][f] for r in (6, 7) for f in range(8)]
    top = [squares[r][f] for r in (0, 1) for f in range(8)]
    bottom_black = sum(1 for s in bottom if s.occupied and s.is_black)
    top_black = sum(1 for s in top if s.occupied and s.is_black)
    if bottom_black == top_black:
        return None
    return bottom_black < top_black


def calibrate_from_image(
    image: np.ndarray,
    classifier: PieceClassifier,
    *,
    corners=None,
    board_px: int = BOARD_PX,
    max_occupancy_mismatch: int = 2,
    train: bool = True,
) -> CalibrationResult:
    """Extract labelled crops from a starting-position photo and train.

    ``max_occupancy_mismatch`` allows a square or two of slack, since a piece
    knocked slightly off-centre can confuse occupancy without invalidating the
    photo. Beyond that we refuse: at that point it is more likely the wrong
    position than a bad reading.
    """
    warnings: list[str] = []

    detection = (
        detect_from_corners(image, corners, board_px)
        if corners is not None
        else detect_board(image, board_px)
    )
    if not detection.confident:
        warnings.append(
            f"Board detection scored {detection.score:.2f}, which is low. "
            "If calibration looks wrong, re-run with manually picked corners."
        )

    reading = analyse(detection.warped)
    reading.corners = detection.corners
    reading.detection_method = detection.method

    turns, mismatch = _best_rotation(reading)
    if mismatch > max_occupancy_mismatch:
        raise CalibrationError(
            f"This does not look like the starting position: {mismatch} squares "
            "disagree with the expected layout even after trying every rotation. "
            "Calibration needs a photo of the board set up for a new game, with "
            "all 32 pieces on their home squares."
        )
    if mismatch:
        warnings.append(
            f"{mismatch} square(s) did not match the starting layout exactly; "
            "training continued, but check the pieces are on their home squares."
        )

    warped = detection.warped
    if turns:
        # Rotate the image itself so crops line up with the rotated grid.
        for _ in range(turns):
            warped = np.rot90(warped, k=-1).copy()
        reading = analyse(warped)
        reading.corners = detection.corners
        reading.detection_method = detection.method

    white_at_bottom = _white_is_at_bottom(reading.squares)
    if white_at_bottom is None:
        warnings.append(
            "Could not tell the light pieces from the dark ones by brightness. "
            "Piece-type training still works, but check your lighting."
        )
    elif not white_at_bottom:
        # Board is the right way up for occupancy but photographed from black's
        # side; two more quarter turns puts white at the bottom.
        for _ in range(2):
            warped = np.rot90(warped, k=-1).copy()
        turns = (turns + 2) % 4
        reading = analyse(warped)
        reading.corners = detection.corners
        reading.detection_method = detection.method

    crops: list[np.ndarray] = []
    labels: list[str] = []
    for r in range(8):
        for f in range(8):
            expected = STARTING_GRID[r][f]
            if not expected:
                continue
            if not reading.squares[r][f].occupied:
                continue  # already counted in the mismatch tally
            crops.append(piece_crop(warped, r, f))
            labels.append(expected.lower())

    if not crops:
        raise CalibrationError("no pieces were found to learn from")

    classifier.add_samples(crops, labels)
    stats = classifier.train() if train else classifier.stats
    if train:
        classifier.save()

    return CalibrationResult(
        samples_added=len(crops),
        rotation_applied=turns,
        stats=stats,
        warnings=warnings,
    )


def add_corrections(
    classifier: PieceClassifier,
    corrections: list[tuple[np.ndarray, str]],
    *,
    retrain: bool = True,
) -> TrainingStats | None:
    """Fold editor corrections back into the training set.

    This is what makes the tool improve with use: every square you fix is a
    labelled example from your own board, your own pieces, and your own
    lighting — which is exactly the distribution that matters.
    """
    if not corrections:
        return classifier.stats
    crops = [crop for crop, _ in corrections]
    labels = [label.lower() for _, label in corrections]
    classifier.add_samples(crops, labels)
    if retrain:
        stats = classifier.train()
        classifier.save()
        return stats
    classifier.save()
    return classifier.stats
