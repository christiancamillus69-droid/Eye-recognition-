"""Shared types and board conventions.

Grid convention, used everywhere in this package: an 8x8 nested list where
``grid[0]`` is rank 8 and ``grid[7]`` is rank 1, and within a rank index 0 is
file a. That is exactly FEN's own reading order, so assembling a FEN is a
straight walk over the grid with no flipping.

A square's piece is a single FEN character (uppercase white, lowercase black),
or the empty string for an empty square.
"""

from __future__ import annotations

from dataclasses import dataclass, field

WHITE_PIECES = "PNBRQK"
BLACK_PIECES = "pnbrqk"
PIECE_CODES = WHITE_PIECES + BLACK_PIECES
PIECE_TYPES = "pnbrqk"  # type only, colour-independent
EMPTY = ""

#: Piece type -> human name, for editor labels and error messages.
PIECE_NAMES = {
    "p": "pawn",
    "n": "knight",
    "b": "bishop",
    "r": "rook",
    "q": "queen",
    "k": "king",
}


def square_name(rank_index: int, file_index: int) -> str:
    """Algebraic name for a grid position, e.g. ``(0, 0)`` -> ``"a8"``."""
    if not (0 <= rank_index < 8 and 0 <= file_index < 8):
        raise ValueError(f"grid position out of range: {(rank_index, file_index)}")
    return f"{'abcdefgh'[file_index]}{8 - rank_index}"


def is_dark_square(rank_index: int, file_index: int) -> bool:
    """True when the board square itself is a dark square.

    a1 (grid position 7,0) is dark, which fixes the parity for the whole board.
    """
    return (rank_index + file_index) % 2 == 1


def empty_grid() -> list[list[str]]:
    """A fresh 8x8 grid of empty squares."""
    return [[EMPTY for _ in range(8)] for _ in range(8)]


@dataclass
class SquareReading:
    """What the pipeline concluded about one square, with its confidences.

    The three stages are kept separate rather than collapsed into a single
    guess, because they have very different reliability: occupancy and colour
    come from geometry and luminance and are usually solid, while ``piece``
    comes from the learned classifier and is the part worth reviewing. The
    editor sorts squares by ``review_priority`` so the least trustworthy
    readings surface first.
    """

    occupied: bool = False
    occupancy_confidence: float = 1.0
    is_black: bool | None = None
    colour_confidence: float = 1.0
    piece: str = EMPTY
    piece_confidence: float = 1.0
    #: Full probability distribution over piece codes, when the classifier
    #: provided one. Used by the editor to offer ranked alternatives.
    alternatives: dict[str, float] = field(default_factory=dict)

    @property
    def review_priority(self) -> float:
        """Lower confidence sorts first. Empty squares are rarely worth review.

        Combines the three stages so that a square which is *probably* empty
        but might not be still gets looked at, rather than being skipped
        because its (nonexistent) piece prediction was confident by default.
        """
        if not self.occupied:
            # An empty square is only interesting if occupancy was borderline.
            return self.occupancy_confidence
        return min(
            self.occupancy_confidence,
            self.colour_confidence,
            self.piece_confidence,
        )


@dataclass
class BoardReading:
    """A complete read of one photograph."""

    squares: list[list[SquareReading]]
    #: Perspective-corrected top-down board image (BGR, as OpenCV returns).
    warped: object | None = None
    #: The four source-image corners used for the warp, in order
    #: top-left, top-right, bottom-right, bottom-left.
    corners: list[tuple[float, float]] | None = None
    #: How the board was located, for the editor to report to the user.
    detection_method: str = "unknown"

    def grid(self) -> list[list[str]]:
        """Just the piece codes, ready for :func:`boardeye.fen.grid_to_fen`."""
        return [[sq.piece for sq in rank] for rank in self.squares]

    def review_order(self) -> list[tuple[int, int]]:
        """Grid positions sorted least-confident first."""
        positions = [(r, f) for r in range(8) for f in range(8)]
        return sorted(positions, key=lambda p: self.squares[p[0]][p[1]].review_priority)
