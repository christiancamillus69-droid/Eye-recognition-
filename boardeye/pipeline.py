"""Photograph in, board reading out — the three stages joined up.

Detection finds the board, square analysis says what is occupied and in which
colour, and the classifier names the piece types. Keeping this orchestration in
one place means the CLI, the editor, and the live scanner all read a board the
same way.
"""

from __future__ import annotations

import numpy as np

from .classifier import PieceClassifier
from .detect import BOARD_PX, Detection, detect_board, detect_from_corners
from .squares import analyse, piece_crop
from .types import BoardReading


def read_position(
    image: np.ndarray,
    classifier: PieceClassifier | None = None,
    *,
    corners=None,
    board_px: int = BOARD_PX,
) -> BoardReading:
    """Read a board from one image.

    Pass ``corners`` to skip automatic detection and use four points the user
    clicked instead. Pass a trained ``classifier`` to get piece types; without
    one you still get a correct occupancy and colour map, which the editor can
    present for manual labelling.
    """
    detection: Detection = (
        detect_from_corners(image, corners, board_px)
        if corners is not None
        else detect_board(image, board_px)
    )

    reading = analyse(detection.warped)
    reading.corners = detection.corners
    reading.detection_method = detection.method

    apply_classifier(reading, classifier)
    return reading


def apply_classifier(
    reading: BoardReading, classifier: PieceClassifier | None
) -> BoardReading:
    """Fill in piece types on an already-analysed board.

    Squares whose type cannot be determined are filled with a pawn at zero
    confidence rather than left empty. A complete-but-flagged board is more
    useful than a board with holes in it: the FEN stays valid, the piece is on
    the right square in the right colour, and the editor sorts zero-confidence
    squares to the front of the review queue so they are the first thing you
    look at.
    """
    if reading.warped is None:
        return reading

    for r in range(8):
        for f in range(8):
            square = reading.squares[r][f]
            if not square.occupied:
                square.piece = ""
                square.piece_confidence = 1.0
                square.alternatives = {}
                continue

            is_black = bool(square.is_black)
            if classifier is None or not classifier.is_trained:
                square.piece = "p" if is_black else "P"
                square.piece_confidence = 0.0
                square.alternatives = {}
                continue

            crop = piece_crop(reading.warped, r, f)
            probabilities = classifier.predict(crop)
            best_type = max(probabilities, key=probabilities.get)
            square.piece = best_type if is_black else best_type.upper()
            square.piece_confidence = float(probabilities[best_type])
            square.alternatives = probabilities

    return reading


def recolour(reading: BoardReading) -> None:
    """Re-apply piece colours to the piece letters after a colour edit."""
    for rank in reading.squares:
        for square in rank:
            if square.occupied and square.piece:
                letter = square.piece.lower()
                square.piece = letter if square.is_black else letter.upper()
