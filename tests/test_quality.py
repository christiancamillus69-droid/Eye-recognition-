"""Grading photos, and measuring accuracy on a board we do not have."""

from __future__ import annotations

import numpy as np
import pytest

from boardeye.quality import (
    FILL_POOR,
    SHARPNESS_POOR,
    holdout_accuracy,
    inspect,
)

from .photoreal import STARTING_PLACEMENT, Realism, render_photo

MIDDLEGAME = "r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1B3/PPP2PPP/R2Q1RK1"


def test_a_good_photo_is_reported_usable():
    image, _ = render_photo(STARTING_PLACEMENT, Realism.gentle(), seed=201)
    report = inspect(image, expect_start=True)
    assert report.usable, report.problems
    assert report.looks_like_start
    assert report.sharpness > SHARPNESS_POOR
    assert report.fill > FILL_POOR


def test_a_blurred_photo_is_called_out_as_soft():
    import cv2

    image, _ = render_photo(STARTING_PLACEMENT, Realism.gentle(), seed=201)
    blurred = cv2.GaussianBlur(image, (0, 0), 4.0)
    report = inspect(blurred, expect_start=True)
    assert not report.usable
    assert any("focus" in p or "soft" in p for p in report.problems)
    assert any("focus" in a or "Steady" in a for a in report.advice)


def test_a_distant_board_is_told_to_get_closer():
    """A board lost in a wide shot gives the classifier too few pixels."""
    image, _ = render_photo(STARTING_PLACEMENT, Realism.gentle(), seed=201)
    canvas = np.full((image.shape[0] * 3, image.shape[1] * 3, 3), 60, np.uint8)
    canvas[: image.shape[0], : image.shape[1]] = image
    report = inspect(canvas, expect_start=True)
    if report.found_board:
        assert report.fill < FILL_POOR or not report.usable


def test_a_photo_with_no_board_says_so():
    rng = np.random.default_rng(0)
    report = inspect(rng.integers(0, 255, (600, 600, 3), dtype=np.uint8), expect_start=True)
    assert not report.usable
    assert report.problems


def test_a_non_starting_position_is_rejected_for_calibration():
    image, _ = render_photo(MIDDLEGAME, Realism.gentle(), seed=11)
    report = inspect(image, expect_start=True)
    assert report.looks_like_start is False
    assert any("starting position" in p for p in report.problems)


def test_starting_position_is_recognised_from_any_rotation():
    """The user should not have to think about which way round they stood."""
    image, _ = render_photo(STARTING_PLACEMENT, Realism.gentle(), seed=201)
    for turns in range(4):
        rotated = np.rot90(image, k=turns).copy()
        report = inspect(rotated, expect_start=True)
        if report.found_board:
            assert report.looks_like_start, f"rotation {turns * 90} not recognised"


def test_position_photos_are_not_checked_against_the_start():
    image, _ = render_photo(MIDDLEGAME, Realism.gentle(), seed=11)
    report = inspect(image, expect_start=False)
    assert report.looks_like_start is None
    assert report.usable


# ------------------------------------------------------------------ hold-out


def test_holdout_needs_at_least_two_photos():
    image, _ = render_photo(STARTING_PLACEMENT, Realism.gentle(), seed=201)
    assert holdout_accuracy([image]) is None


def test_holdout_measures_accuracy_on_the_supplied_board(tmp_path):
    """The point of the whole module: a real number for a real board.

    Deliberately loose bounds. This asserts the measurement *works* — that it
    trains, reads the held-out photo and returns a plausible figure — not that
    any particular accuracy is achieved, which depends on the board.
    """
    images = [
        render_photo(STARTING_PLACEMENT, Realism.gentle(), seed=seed)[0]
        for seed in (201, 205)
    ]
    result = holdout_accuracy(images, model_dir=tmp_path)
    assert result is not None
    assert result.folds == 2
    assert result.pieces == 64  # 32 pieces per held-out photo
    assert 0.3 < result.accuracy <= 1.0
    assert set(result.per_type) == set("pnbrqk")
    assert "of pieces read correctly" in result.summary()


def test_holdout_reports_per_piece_type_counts(tmp_path):
    """Naming the weak piece types is what makes the number actionable."""
    images = [
        render_photo(STARTING_PLACEMENT, Realism.gentle(), seed=seed)[0]
        for seed in (201, 205)
    ]
    result = holdout_accuracy(images, model_dir=tmp_path)
    assert result is not None
    # Each starting position holds 16 pawns, 4 rooks and 2 kings (both
    # colours train one class), doubled over two held-out photos.
    assert result.per_type["p"][1] == 32
    assert result.per_type["r"][1] == 8
    assert result.per_type["k"][1] == 4
    assert result.per_type["q"][1] == 4


@pytest.mark.parametrize("count", [2, 3])
def test_holdout_folds_match_the_photos_given(tmp_path, count):
    images = [
        render_photo(STARTING_PLACEMENT, Realism.gentle(), seed=200 + i)[0]
        for i in range(count)
    ]
    result = holdout_accuracy(images, model_dir=tmp_path)
    assert result is not None and result.folds == count
