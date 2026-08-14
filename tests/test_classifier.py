"""The learned half: calibration, recognition, confidence, and persistence.

These tests validate the *mechanism* on synthetic pieces. They deliberately do
not claim anything about accuracy on real wooden pieces under real light —
only a photograph of an actual board can establish that, and that test is the
user's to run.
"""

from __future__ import annotations

import numpy as np
import pytest

from boardeye.calibrate import (
    CalibrationError,
    add_corrections,
    calibrate_from_image,
)
from boardeye.classifier import PieceClassifier, augment, extract_features, piece_map
from boardeye.fen import placement_to_grid
from boardeye.pipeline import read_position
from boardeye.squares import piece_crop
from boardeye.types import square_name

from .conftest import POSITIONS
from .render import STARTING_FEN, render_board, render_photo


# ---------------------------------------------------------------- features


def test_piece_map_discards_the_pieces_own_colour():
    """The map's bright region must mark where the piece is, whatever its shade.

    This is what allows a white rook and a black rook to train one class. It is
    a weaker claim than "the maps are identical" — see the transfer test below
    for how far the invariance actually reaches.
    """
    for piece in ("R", "r"):
        board = render_board(f"8/8/8/3{piece}4/8/8/8/8", square_px=64)
        field = piece_map(piece_crop(board, 3, 3))
        empty = piece_map(piece_crop(render_board("8/8/8/8/8/8/8/8", square_px=64), 3, 3))
        assert field.mean() > empty.mean() * 2, f"{piece}: piece did not stand out"


def test_cross_colour_transfer_is_real_but_partial(tmp_path):
    """Train on white pieces only, read black ones: better than chance, not solved.

    Pinned deliberately. The six-class design is justified by both colours
    training one class, *not* by one colour generalising to the other — and
    conflating those would make it tempting to calibrate from a half-board.
    The lower bound guards the design; the upper bound documents that the
    invariance is incomplete, so calibration must keep showing both colours.
    """
    from boardeye.calibrate import STARTING_GRID
    from boardeye.detect import detect_board

    photo, _ = render_photo(STARTING_FEN, tilt=0.10, blur=0.5, noise=0.01, seed=31)
    warped = detect_board(photo).warped

    def crops(ranks):
        return (
            [piece_crop(warped, r, f) for r in ranks for f in range(8)],
            [STARTING_GRID[r][f].lower() for r in ranks for f in range(8)],
        )

    classifier = PieceClassifier(tmp_path)
    white_crops, white_labels = crops([6, 7])
    classifier.add_samples(white_crops, white_labels)
    classifier.train()

    black_crops, black_labels = crops([0, 1])
    correct = sum(
        max(classifier.predict(crop), key=classifier.predict(crop).get) == label
        for crop, label in zip(black_crops, black_labels)
    )
    accuracy = correct / len(black_labels)
    assert accuracy > 1 / 6, "no better than guessing — the map is not colour-agnostic at all"
    assert accuracy < 1.0, "if this ever passes fully, revisit the calibration advice"


def test_features_are_finite_and_fixed_length():
    board = render_board(STARTING_FEN)
    lengths = {len(extract_features(piece_crop(board, 0, f))) for f in range(8)}
    assert len(lengths) == 1
    assert np.isfinite(extract_features(piece_crop(board, 0, 0))).all()


def test_augmentation_preserves_shape_and_adds_variety():
    board = render_board(STARTING_FEN)
    crop = piece_crop(board, 0, 0)
    variants = augment(crop)
    assert len(variants) > 8
    assert all(v.shape == crop.shape for v in variants)
    # The mirror must actually differ, or augmentation is a no-op.
    assert any(not np.array_equal(v, crop) for v in variants)


# ------------------------------------------------------------- calibration


def test_calibration_learns_all_six_piece_types(tmp_path):
    classifier = PieceClassifier(tmp_path)
    photo, _ = render_photo(STARTING_FEN, tilt=0.12, blur=0.6, noise=0.01, seed=21)
    result = calibrate_from_image(photo, classifier)
    assert result.samples_added == 32
    assert set(classifier.class_counts()) == set("pnbrqk")
    assert classifier.class_counts()["p"] == 16


def test_calibration_rejects_a_non_starting_position(tmp_path):
    classifier = PieceClassifier(tmp_path)
    photo, _ = render_photo(POSITIONS["middlegame"], tilt=0.1, seed=22)
    with pytest.raises(CalibrationError, match="starting position"):
        calibrate_from_image(photo, classifier)


def test_calibration_corrects_a_board_photographed_from_blacks_side(tmp_path):
    """The starting position looks identical upside down; colour resolves it."""
    classifier = PieceClassifier(tmp_path)
    photo, _ = render_photo(STARTING_FEN, tilt=0.10, blur=0.5, noise=0.01, seed=23)
    upside_down = np.rot90(photo, k=2).copy()
    result = calibrate_from_image(upside_down, classifier)
    assert result.rotation_applied == 2
    assert result.samples_added == 32


def test_calibration_survives_a_quarter_turn(tmp_path):
    classifier = PieceClassifier(tmp_path)
    photo, _ = render_photo(STARTING_FEN, tilt=0.0, blur=0.5, noise=0.01, seed=24)
    turned = np.rot90(photo, k=-1).copy()
    result = calibrate_from_image(turned, classifier)
    assert result.rotation_applied in (1, 3)
    assert result.samples_added == 32


def test_untrained_classifier_refuses_to_predict(tmp_path):
    classifier = PieceClassifier(tmp_path)
    board = render_board(STARTING_FEN)
    with pytest.raises(RuntimeError, match="calibrate"):
        classifier.predict(piece_crop(board, 0, 0))


def test_training_needs_more_than_one_class(tmp_path):
    classifier = PieceClassifier(tmp_path)
    board = render_board(STARTING_FEN)
    classifier.add_samples([piece_crop(board, 1, 0)], ["p"])
    with pytest.raises(ValueError, match="two different piece types"):
        classifier.train()


def test_rejects_a_label_that_is_not_a_piece(tmp_path):
    classifier = PieceClassifier(tmp_path)
    board = render_board(STARTING_FEN)
    with pytest.raises(ValueError):
        classifier.add_samples([piece_crop(board, 0, 0)], ["x"])


# ------------------------------------------------------------- recognition


#: Positions with enough pieces for an accuracy figure to mean something.
SUBSTANTIAL = {
    name: placement
    for name, placement in POSITIONS.items()
    if sum(bool(cell) for rank in placement_to_grid(placement) for cell in rank) >= 7
}


@pytest.mark.parametrize("name,placement", sorted(SUBSTANTIAL.items()))
def test_reads_positions_accurately(trained_classifier, scan_photo, name, placement):
    photo, _ = scan_photo(placement)
    reading = read_position(photo, trained_classifier)
    truth = placement_to_grid(placement)
    got = reading.grid()

    pieces = sum(1 for r in range(8) for f in range(8) if truth[r][f])
    wrong = [
        f"{square_name(r, f)} {truth[r][f]}->{got[r][f] or 'empty'}"
        for r in range(8)
        for f in range(8)
        if truth[r][f] != got[r][f]
    ]
    accuracy = (pieces - len(wrong)) / pieces
    assert accuracy >= 0.85, f"{name}: {accuracy:.0%} correct, wrong: {wrong}"


def test_very_sparse_boards_place_pieces_right_but_may_misname_them(
    trained_classifier, scan_photo
):
    """A known limitation, pinned so it cannot regress silently.

    On a two-piece board the classifier has no context and a lone king is
    readily read as a queen. What must still hold is everything the
    non-learned half promises — the pieces are on the correct squares in the
    correct colours — and that every piece is flagged, so the two keystrokes
    needed to fix it are actually prompted for.
    """
    placement = "8/8/4k3/8/8/2K5/8/8"
    photo, _ = scan_photo(placement)
    reading = read_position(photo, trained_classifier)
    truth = placement_to_grid(placement)

    occupied = [(r, f) for r in range(8) for f in range(8) if reading.squares[r][f].occupied]
    assert occupied == [(r, f) for r in range(8) for f in range(8) if truth[r][f]]
    for r, f in occupied:
        assert reading.squares[r][f].piece.islower() == truth[r][f].islower()

    # With so few pieces every one of them sits at the top of the review queue.
    order = [p for p in reading.review_order() if reading.squares[p[0]][p[1]].occupied]
    assert set(order) == set(occupied)


def test_colour_is_never_wrong_even_when_type_is(trained_classifier, scan_photo):
    """Colour comes from clustering, not the model, and should be solid."""
    photo, _ = scan_photo(POSITIONS["middlegame"])
    reading = read_position(photo, trained_classifier)
    truth = placement_to_grid(POSITIONS["middlegame"])
    for r in range(8):
        for f in range(8):
            if truth[r][f] and reading.squares[r][f].piece:
                expected_black = truth[r][f].islower()
                assert reading.squares[r][f].piece.islower() == expected_black


def test_mistakes_surface_near_the_front_of_the_review_queue(
    trained_classifier, scan_photo
):
    """The whole review design assumes errors cluster among low-confidence squares."""
    photo, _ = scan_photo(POSITIONS["tactical"])
    reading = read_position(photo, trained_classifier)
    truth = placement_to_grid(POSITIONS["tactical"])
    order = reading.review_order()

    occupied_order = [(r, f) for r, f in order if reading.squares[r][f].occupied]
    ranks = [
        occupied_order.index((r, f))
        for r in range(8)
        for f in range(8)
        if truth[r][f] != reading.grid()[r][f] and (r, f) in occupied_order
    ]
    if not ranks:
        return  # nothing wrong is a fine outcome
    assert min(ranks) < len(occupied_order) // 2, (
        "every mistake sat in the more-confident half of the queue, which would "
        "defeat confidence-ordered review"
    )


def test_without_a_model_pieces_are_placed_but_marked_unknown(scan_photo, tmp_path):
    """An untrained run should still produce a complete, correctable board."""
    photo, _ = scan_photo(POSITIONS["middlegame"])
    reading = read_position(photo, PieceClassifier(tmp_path))
    truth = placement_to_grid(POSITIONS["middlegame"])
    for r in range(8):
        for f in range(8):
            square = reading.squares[r][f]
            assert square.occupied == bool(truth[r][f])
            if square.occupied:
                assert square.piece.lower() == "p"
                assert square.piece_confidence == 0.0


# ------------------------------------------------------------- persistence


def test_model_round_trips_through_disk(tmp_path, scan_photo):
    original = PieceClassifier(tmp_path)
    photo, _ = render_photo(STARTING_FEN, tilt=0.1, blur=0.5, noise=0.01, seed=25)
    calibrate_from_image(photo, original)

    reloaded = PieceClassifier(tmp_path)
    assert reloaded.load() is True
    assert reloaded.sample_count == original.sample_count

    scan, _ = scan_photo(POSITIONS["endgame"])
    assert read_position(scan, reloaded).grid() == read_position(scan, original).grid()


def test_corrections_are_added_and_change_the_dataset(tmp_path):
    classifier = PieceClassifier(tmp_path)
    photo, _ = render_photo(STARTING_FEN, tilt=0.1, blur=0.5, noise=0.01, seed=26)
    calibrate_from_image(photo, classifier)
    before = classifier.sample_count

    board = render_board("8/8/8/3q4/8/8/8/8", square_px=64)
    add_corrections(classifier, [(piece_crop(board, 3, 3), "q")])
    assert classifier.sample_count == before + 1
    assert classifier.class_counts()["q"] == 3


def test_adding_no_corrections_is_harmless(tmp_path):
    classifier = PieceClassifier(tmp_path)
    photo, _ = render_photo(STARTING_FEN, tilt=0.1, seed=27)
    calibrate_from_image(photo, classifier)
    before = classifier.sample_count
    add_corrections(classifier, [])
    assert classifier.sample_count == before
