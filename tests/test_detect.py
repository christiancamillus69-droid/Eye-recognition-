"""Board localisation: does it find the board, and does it know when it hasn't?"""

from __future__ import annotations

import numpy as np
import pytest

from boardeye.detect import (
    BoardNotFound,
    MIN_ACCEPTABLE_SCORE,
    checkerboard_score,
    detect_board,
    detect_from_corners,
    order_corners,
    warp_from_corners,
)

from .render import STARTING_FEN, render_board, render_photo

MIDDLEGAME = "r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1B3/PPP2PPP/R2Q1RK1"


def max_corner_error(found, expected) -> float:
    """Largest distance between corresponding corners.

    Both sequences are ordered top-left, top-right, bottom-right, bottom-left —
    the detector guarantees it and the renderer produces it — so corners are
    compared position by position. Sorting them first would pair a corner with
    the wrong one and report a meaningless distance.
    """
    return max(
        float(np.hypot(f[0] - e[0], f[1] - e[1])) for f, e in zip(found, expected)
    )


def test_order_corners_puts_them_clockwise_from_top_left():
    scrambled = np.float32([[100, 900], [900, 100], [100, 100], [900, 900]])
    ordered = order_corners(scrambled)
    assert list(ordered[0]) == [100, 100]  # top-left
    assert list(ordered[1]) == [900, 100]  # top-right
    assert list(ordered[2]) == [900, 900]  # bottom-right
    assert list(ordered[3]) == [100, 900]  # bottom-left


def test_a_rendered_board_scores_near_perfect():
    assert checkerboard_score(render_board(STARTING_FEN)) > 0.95


@pytest.mark.parametrize(
    "placement,tilt",
    [
        (STARTING_FEN, 0.0),
        (STARTING_FEN, 0.18),
        (MIDDLEGAME, 0.22),
        (MIDDLEGAME, 0.30),
        ("8/8/4k3/8/8/2K5/8/8", 0.15),  # nearly empty board
    ],
)
def test_detects_board_across_tilts_and_positions(placement, tilt):
    photo, corners = render_photo(placement, tilt=tilt, blur=1.0, noise=0.01, seed=1)
    detection = detect_board(photo)
    assert detection.confident
    assert max_corner_error(detection.corners, corners) < 12.0


def test_warp_is_square_and_aligned_to_the_grid():
    photo, _ = render_photo(MIDDLEGAME, tilt=0.2, seed=2)
    warped = detect_board(photo).warped
    assert warped.shape[0] == warped.shape[1]
    # If the warp were misaligned, square medians would stop alternating.
    assert checkerboard_score(warped) > 0.9


@pytest.mark.parametrize(
    "image",
    [
        np.random.default_rng(0).integers(0, 255, (600, 600, 3), dtype=np.uint8),
        np.full((600, 600, 3), 128, np.uint8),
    ],
    ids=["random-noise", "flat-grey"],
)
def test_non_boards_are_not_accepted(image):
    """The score is the trust signal; it must not endorse a non-board."""
    try:
        detection = detect_board(image)
    except BoardNotFound:
        return  # refusing outright is the strongest possible answer
    assert not detection.confident
    assert detection.score < MIN_ACCEPTABLE_SCORE


def test_empty_image_raises():
    with pytest.raises(BoardNotFound):
        detect_board(np.zeros((0, 0, 3), np.uint8))


def test_manual_corners_bypass_detection():
    photo, corners = render_photo(MIDDLEGAME, tilt=0.25, seed=4)
    detection = detect_from_corners(photo, corners)
    assert detection.method == "manual"
    assert detection.score > 0.9


def test_warp_from_corners_is_order_independent():
    photo, corners = render_photo(MIDDLEGAME, tilt=0.2, seed=5)
    rotated_input = corners[2:] + corners[:2]
    assert np.array_equal(
        warp_from_corners(photo, corners), warp_from_corners(photo, rotated_input)
    )


def test_refinement_does_not_make_things_worse():
    photo, _ = render_photo(MIDDLEGAME, tilt=0.2, blur=1.0, seed=6)
    assert detect_board(photo, refine=True).score >= detect_board(photo, refine=False).score
