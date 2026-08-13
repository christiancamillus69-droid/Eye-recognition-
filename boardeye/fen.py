"""Turn an 8x8 grid into a FEN, and check that the result is a sane position.

A photograph tells you where the pieces are. It cannot tell you whose move it
is, whether either side may still castle, or whether an en passant capture is
available — those are facts about the game's history, not its current picture.
So they are defaults here that the editor lets you override, and the castling
default is an inference from piece placement rather than a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

from .types import EMPTY, PIECE_CODES, empty_grid

#: Grid positions of the squares castling rights depend on.
_E1, _H1, _A1 = (7, 4), (7, 7), (7, 0)
_E8, _H8, _A8 = (0, 4), (0, 7), (0, 0)


def infer_castling(grid: list[list[str]]) -> str:
    """Best-effort castling rights from piece placement alone.

    A king and rook sitting on their home squares is consistent with the right
    still being available, so we assume it is. That is a guess, and it is
    sometimes wrong — a king that moved and came home keeps no rights. The
    editor exposes these as checkboxes precisely because the photograph cannot
    settle the question.
    """
    rights = ""
    if grid[_E1[0]][_E1[1]] == "K":
        if grid[_H1[0]][_H1[1]] == "R":
            rights += "K"
        if grid[_A1[0]][_A1[1]] == "R":
            rights += "Q"
    if grid[_E8[0]][_E8[1]] == "k":
        if grid[_H8[0]][_H8[1]] == "r":
            rights += "k"
        if grid[_A8[0]][_A8[1]] == "r":
            rights += "q"
    return rights or "-"


def grid_to_placement(grid: list[list[str]]) -> str:
    """The piece-placement field of a FEN (the part before the first space)."""
    ranks = []
    for rank in grid:
        out: list[str] = []
        run = 0
        for piece in rank:
            if piece == EMPTY:
                run += 1
                continue
            if run:
                out.append(str(run))
                run = 0
            if piece not in PIECE_CODES:
                raise ValueError(f"not a FEN piece character: {piece!r}")
            out.append(piece)
        if run:
            out.append(str(run))
        ranks.append("".join(out))
    return "/".join(ranks)


def placement_to_grid(placement: str) -> list[list[str]]:
    """Inverse of :func:`grid_to_placement`."""
    grid = empty_grid()
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError(f"expected 8 ranks, got {len(ranks)}")
    for r, rank in enumerate(ranks):
        f = 0
        for ch in rank:
            if ch.isdigit():
                f += int(ch)
            elif ch in PIECE_CODES:
                if f >= 8:
                    raise ValueError(f"rank {8 - r} overflows past file h")
                grid[r][f] = ch
                f += 1
            else:
                raise ValueError(f"unexpected character in FEN rank: {ch!r}")
        if f != 8:
            raise ValueError(f"rank {8 - r} describes {f} files, expected 8")
    return grid


def rotate_grid(grid: list[list[str]], quarter_turns: int = 1) -> list[list[str]]:
    """Rotate the board clockwise by 90 degrees, ``quarter_turns`` times.

    Board orientation is not recoverable from a photograph — nothing in the
    image says which edge white sat behind — so the editor offers a rotate
    button and this is what backs it.
    """
    out = [row[:] for row in grid]
    for _ in range(quarter_turns % 4):
        # Clockwise: new[r][f] = old[7-f][r]
        out = [[out[7 - f][r] for f in range(8)] for r in range(8)]
    return out


@dataclass
class Position:
    """A grid plus the game-state facts a photograph cannot supply."""

    grid: list[list[str]] = field(default_factory=empty_grid)
    side_to_move: str = "w"
    castling: str | None = None  # None means "infer from placement"
    en_passant: str = "-"
    halfmove_clock: int = 0
    fullmove_number: int = 1

    def fen(self) -> str:
        if self.side_to_move not in ("w", "b"):
            raise ValueError(f"side_to_move must be 'w' or 'b', got {self.side_to_move!r}")
        castling = self.castling if self.castling is not None else infer_castling(self.grid)
        return " ".join(
            [
                grid_to_placement(self.grid),
                self.side_to_move,
                castling or "-",
                self.en_passant or "-",
                str(self.halfmove_clock),
                str(self.fullmove_number),
            ]
        )

    @classmethod
    def from_fen(cls, fen: str) -> "Position":
        parts = fen.split()
        if len(parts) < 2:
            raise ValueError("FEN needs at least a placement and a side to move")
        placement, side = parts[0], parts[1]
        castling = parts[2] if len(parts) > 2 else "-"
        en_passant = parts[3] if len(parts) > 3 else "-"
        halfmove = int(parts[4]) if len(parts) > 4 else 0
        fullmove = int(parts[5]) if len(parts) > 5 else 1
        return cls(
            grid=placement_to_grid(placement),
            side_to_move=side,
            castling=castling,
            en_passant=en_passant,
            halfmove_clock=halfmove,
            fullmove_number=fullmove,
        )


#: python-chess status flags mapped to messages worth showing a chess player.
#: Anything not listed is reported generically rather than silently dropped.
_STATUS_MESSAGES: list[tuple[int, str]] = [
    (chess.STATUS_NO_WHITE_KING, "No white king on the board."),
    (chess.STATUS_NO_BLACK_KING, "No black king on the board."),
    (chess.STATUS_TOO_MANY_KINGS, "More than one king for a side."),
    (chess.STATUS_TOO_MANY_WHITE_PAWNS, "More than 8 white pawns."),
    (chess.STATUS_TOO_MANY_BLACK_PAWNS, "More than 8 black pawns."),
    (chess.STATUS_PAWNS_ON_BACKRANK, "A pawn is on rank 1 or rank 8, which is impossible."),
    (chess.STATUS_TOO_MANY_WHITE_PIECES, "Too many white pieces."),
    (chess.STATUS_TOO_MANY_BLACK_PIECES, "Too many black pieces."),
    (chess.STATUS_BAD_CASTLING_RIGHTS, "Castling rights do not match where the king and rooks are."),
    (chess.STATUS_INVALID_EP_SQUARE, "The en passant square is not reachable from this position."),
    (
        chess.STATUS_OPPOSITE_CHECK,
        "The side not to move is in check, so this position cannot arise. "
        "Check the side-to-move setting, or look for a misread piece.",
    ),
    (chess.STATUS_RACE_CHECK, "Both kings appear to be in check."),
    (chess.STATUS_RACE_OVER, "Position is inconsistent (racing-kings check)."),
    (chess.STATUS_RACE_MATERIAL, "Position is inconsistent (racing-kings material)."),
    (chess.STATUS_EMPTY, "The board is empty."),
]


def validate(fen: str) -> list[str]:
    """Return human-readable problems with a FEN. Empty list means it is fine.

    This warns rather than raises: an engine will happily analyse many
    positions python-chess considers irregular, and a player who set up a
    puzzle deliberately should not be blocked. The one genuinely useful catch
    is ``OPPOSITE_CHECK``, which almost always means either the side-to-move
    toggle is wrong or a piece was misread — both worth flagging before you
    trust an evaluation.
    """
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        return [f"Not a valid FEN: {exc}"]

    status = board.status()
    if status == chess.STATUS_VALID:
        return []

    problems = [message for flag, message in _STATUS_MESSAGES if status & flag]
    # Make sure an unrecognised flag still surfaces instead of vanishing.
    covered = 0
    for flag, _ in _STATUS_MESSAGES:
        covered |= flag
    if status & ~covered:
        problems.append("The position is irregular in a way python-chess flagged.")
    return problems


#: The standard complement of each piece type per side.
_STANDARD = {"p": 8, "n": 2, "b": 2, "r": 2, "q": 1, "k": 1}


def material_warnings(grid: list[list[str]]) -> list[str]:
    """Flag piece counts that no legal game could produce.

    This catches a failure mode confidence scores cannot: a classifier that is
    *confidently* wrong. A single misread piece is invisible here, but the
    characteristic bad read — one piece type bleeding into another across the
    board, so that eight pawns come back as eight rooks — produces material
    that is instantly impossible, and saying so is far more useful than a
    per-square probability.

    Extra pieces are only legal via promotion, and every promotion costs a
    pawn, so a side's pawns plus its promoted extras can never exceed eight.
    """
    problems: list[str] = []
    for is_white, label in ((True, "White"), (False, "Black")):
        counts = {piece: 0 for piece in _STANDARD}
        for rank in grid:
            for cell in rank:
                if cell and cell.isupper() == is_white:
                    counts[cell.lower()] += 1

        if counts["p"] > 8:
            problems.append(f"{label} has {counts['p']} pawns; 8 is the maximum.")
        if counts["k"] > 1:
            problems.append(f"{label} has {counts['k']} kings.")

        extras = sum(
            max(0, counts[piece] - _STANDARD[piece]) for piece in ("n", "b", "r", "q")
        )
        surplus = [
            f"{counts[piece]} {piece.upper()}s"
            for piece in ("n", "b", "r", "q")
            if counts[piece] > _STANDARD[piece]
        ]
        if extras and counts["p"] + extras > 8:
            problems.append(
                f"{label} has {' and '.join(surplus)} alongside {counts['p']} pawns. "
                "Extra pieces require promotions, and there are not enough missing "
                "pawns to account for them — most likely a piece was misread."
            )
        elif any(counts[piece] >= 3 for piece in ("n", "b", "r", "q")):
            # Reachable by promotion, so not an error — but three of a kind is
            # rare enough in a real game that it is far more often a misread,
            # and this is the shape the bad reads take: one piece type
            # swallowing another across the board.
            problems.append(
                f"{label} has {' and '.join(surplus)}. That is possible after "
                "promotions but unusual — worth confirming those pieces are right."
            )
    return problems


def is_legal_setup(fen: str) -> bool:
    """True when python-chess considers the position fully valid."""
    try:
        return chess.Board(fen).status() == chess.STATUS_VALID
    except ValueError:
        return False
