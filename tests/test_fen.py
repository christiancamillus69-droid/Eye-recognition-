"""FEN assembly, castling inference, rotation, validation, and handoff formats."""

from __future__ import annotations

import chess
import pytest

from boardeye import export
from boardeye.fen import (
    Position,
    grid_to_placement,
    infer_castling,
    is_legal_setup,
    material_warnings,
    placement_to_grid,
    rotate_grid,
    validate,
)
from boardeye.types import empty_grid, is_dark_square, square_name

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
MIDDLEGAME = "r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1B3/PPP2PPP/R2Q1RK1"


@pytest.mark.parametrize("placement", [START, MIDDLEGAME, "8/5pk1/6p1/8/1P6/2K5/5PP1/8", "8/8/8/8/8/8/8/8"])
def test_placement_round_trips(placement):
    assert grid_to_placement(placement_to_grid(placement)) == placement


def test_grid_convention_matches_python_chess():
    """Our grid[0] is rank 8; confirm that against an independent library."""
    grid = placement_to_grid(START)
    board = chess.Board(f"{START} w KQkq - 0 1")
    for rank_index in range(8):
        for file_index in range(8):
            square = chess.square(file_index, 7 - rank_index)
            piece = board.piece_at(square)
            expected = piece.symbol() if piece else ""
            assert grid[rank_index][file_index] == expected, square_name(rank_index, file_index)


def test_a1_is_a_dark_square():
    assert is_dark_square(7, 0) is True   # a1
    assert is_dark_square(7, 1) is False  # b1
    assert is_dark_square(0, 0) is False  # a8


@pytest.mark.parametrize(
    "placement,expected",
    [
        (START, "KQkq"),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR", "Kkq"),  # white a1 rook gone
        ("rnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", "KQq"),  # black h8 rook gone
        ("1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", "KQk"),  # black a8 rook gone
        ("rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1BNR", "-"),    # neither king home
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBN1", "kq"),   # both white rooks gone
    ],
)
def test_castling_inference(placement, expected):
    assert infer_castling(placement_to_grid(placement)) == expected


def test_position_builds_the_standard_start_fen():
    position = Position(grid=placement_to_grid(START))
    assert position.fen() == f"{START} w KQkq - 0 1"


def test_position_round_trips_through_fen():
    original = Position(
        grid=placement_to_grid(MIDDLEGAME),
        side_to_move="b",
        castling="-",
        en_passant="e3",
        halfmove_clock=3,
        fullmove_number=17,
    )
    assert Position.from_fen(original.fen()).fen() == original.fen()


def test_explicit_castling_overrides_inference():
    grid = placement_to_grid(START)
    assert Position(grid=grid, castling="-").fen().split()[2] == "-"


def test_rotation_is_a_quarter_turn_group():
    grid = placement_to_grid(MIDDLEGAME)
    assert rotate_grid(grid, 4) == grid
    assert rotate_grid(grid, 1) != grid
    assert rotate_grid(rotate_grid(grid, 1), 3) == grid


def test_rotation_moves_a1_corner_piece_as_expected():
    grid = empty_grid()
    grid[7][0] = "R"  # a1
    # One clockwise turn takes a1 to a8, which is grid[0][0].
    assert rotate_grid(grid, 1)[0][0] == "R"


def test_side_to_move_must_be_valid():
    with pytest.raises(ValueError):
        Position(grid=placement_to_grid(START), side_to_move="x").fen()


def test_bad_piece_character_is_rejected():
    grid = empty_grid()
    grid[0][0] = "X"
    with pytest.raises(ValueError):
        grid_to_placement(grid)


@pytest.mark.parametrize("bad", ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP", "9/8/8/8/8/8/8/8", "rnbqkbnX/8/8/8/8/8/8/8"])
def test_malformed_placements_raise(bad):
    with pytest.raises(ValueError):
        placement_to_grid(bad)


# ------------------------------------------------------------------ validation


def test_starting_position_validates_clean():
    assert validate(f"{START} w KQkq - 0 1") == []
    assert is_legal_setup(f"{START} w KQkq - 0 1")


def test_opposite_check_is_reported():
    """The single most useful catch: it means a misread or a wrong side-to-move."""
    problems = validate("4k3/8/8/8/8/8/8/4K2r b - - 0 1")
    assert problems and "not to move is in check" in problems[0]


def test_missing_king_is_reported():
    assert any("king" in p.lower() for p in validate("8/8/8/8/8/8/8/4K3 w - - 0 1"))


def test_unparseable_fen_is_reported_not_raised():
    problems = validate("not a fen at all")
    assert len(problems) == 1 and "Not a valid FEN" in problems[0]


# -------------------------------------------------------------------- material


@pytest.mark.parametrize("placement", [START, MIDDLEGAME, "8/5pk1/6p1/8/1P6/2K5/5PP1/8"])
def test_real_positions_raise_no_material_warning(placement):
    assert material_warnings(placement_to_grid(placement)) == []


def test_promotion_to_a_second_queen_is_not_flagged():
    """Two queens is ordinary; warning about it would train users to ignore warnings."""
    assert material_warnings(placement_to_grid("rnbqkbnr/1ppppppp/8/8/8/8/PPPPPPP1/RNBQKBNQ")) == []


def test_impossible_pawn_count_is_flagged():
    problems = material_warnings(placement_to_grid("rnbqkbnr/pppppppp/p7/8/8/8/PPPPPPPP/RNBQKBNR"))
    assert any("9 pawns" in p for p in problems)


def test_extra_pieces_without_missing_pawns_are_flagged():
    problems = material_warnings(placement_to_grid("rnbqkbnr/pppppppp/8/8/8/1B6/PPPPPPPP/RNBQKBNR"))
    assert any("promotions" in p for p in problems)


def test_the_characteristic_misread_is_flagged():
    """One piece type swallowing another is the bad read this must catch."""
    swallowed = "r1bq1rk1/pr2prbr/2np1nr1/8/2BNR3/2N1B3/RPR2PRP/R2Q1RK1"
    assert material_warnings(placement_to_grid(swallowed))


# ---------------------------------------------------------------------- export


def test_lichess_url_uses_underscores_and_keeps_rank_slashes():
    url = export.lichess_url(f"{START} w KQkq - 0 1")
    assert url == f"https://lichess.org/analysis/{START}_w_KQkq_-_0_1"
    assert " " not in url


def test_chesscom_url_percent_encodes_the_fen():
    url = export.chesscom_url(f"{START} w KQkq - 0 1")
    assert "%2F" in url and " " not in url


def test_pgn_carries_setup_and_fen_tags():
    fen = f"{MIDDLEGAME} w - - 0 1"
    pgn = export.to_pgn(fen)
    assert '[SetUp "1"]' in pgn
    assert f'[FEN "{fen}"]' in pgn


def test_pgn_is_readable_by_python_chess():
    """The point of the PGN is that other software loads it, so check that."""
    import io

    import chess.pgn

    fen = f"{MIDDLEGAME} w - - 0 1"
    game = chess.pgn.read_game(io.StringIO(export.to_pgn(fen)))
    assert game is not None
    assert game.board().fen() == fen


def test_build_produces_every_handoff_format():
    handoff = export.build(f"{START} w KQkq - 0 1")
    assert set(handoff.as_dict()) == {"fen", "lichess", "chesscom", "pgn"}
