"""Tests against images that behave like photographs rather than diagrams.

The flat renderer in `render.py` gave the pipeline an easy time: 2-D glyphs,
flat colour, even light, no shadows, a plain backdrop. Measured side by side,
it overstated end-to-end accuracy by roughly ten points. These tests use
`photoreal.py` instead — pieces that stand up and overhang the square behind,
cast shadows, wood grain, uneven light, table clutter, camera roll and JPEG
artefacts — so the thresholds below are the ones worth defending.

They are slower than the rest of the suite because each image is rendered and
then genuinely searched for a board.
"""

from __future__ import annotations

import numpy as np
import pytest

from boardeye.calibrate import calibrate_from_image
from boardeye.classifier import PieceClassifier
from boardeye.detect import MIN_ACCEPTABLE_SCORE, BoardNotFound, detect_board
from boardeye.fen import placement_to_grid
from boardeye.pipeline import read_position
from boardeye.squares import analyse

from .photoreal import STARTING_PLACEMENT, Realism, render_photo

MIDDLEGAME = "r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1B3/PPP2PPP/R2Q1RK1"
ENDGAME = "8/5pk1/6p1/8/1P6/2K5/5PP1/8"

#: Two calibration photos, as the README recommends. See
#: :func:`test_a_second_calibration_photo_rescues_a_bad_first_one` for what
#: that recommendation is actually based on — quality matters far more than
#: count, and the second photo is insurance rather than improvement.
CALIBRATION = [(201, "gentle"), (202, "typical")]


@pytest.fixture(scope="module")
def photo_classifier(tmp_path_factory) -> PieceClassifier:
    classifier = PieceClassifier(tmp_path_factory.mktemp("photo-model"))
    for seed, level in CALIBRATION:
        image, _ = render_photo(
            STARTING_PLACEMENT, getattr(Realism, level)(), seed=seed
        )
        calibrate_from_image(image, classifier)
    return classifier


# ----------------------------------------------------------------- detection


@pytest.mark.parametrize("level", ["gentle", "typical"])
def test_board_is_found_in_photo_like_images(level):
    """Everyday conditions must not need the manual-corner fallback."""
    accepted = 0
    for seed in range(6):
        image, _ = render_photo(MIDDLEGAME, getattr(Realism, level)(), seed=seed)
        try:
            accepted += detect_board(image).confident
        except BoardNotFound:
            pass
    assert accepted >= 5, f"{level}: only {accepted}/6 detections were trusted"


def test_harsh_conditions_either_work_or_admit_defeat():
    """A bad photo may fail; what it must not do is fail while claiming success.

    Judged on occupancy damage rather than corner pixels, deliberately.
    Distance from the true corners turns out to be a poor stand-in for what a
    user experiences: a detection can sit 180px out — a whole square — and
    still cost only a handful of wrong squares, because the grid it locked
    onto is still a grid. Wrong squares are what force corrections, so wrong
    squares are what the bound is written against.
    """
    truth = placement_to_grid(MIDDLEGAME)
    damage = []
    for seed in range(8):
        image, _ = render_photo(MIDDLEGAME, Realism.harsh(), seed=seed)
        try:
            detection = detect_board(image)
        except BoardNotFound:
            continue
        if not detection.confident:
            continue  # admitting defeat is always allowed
        reading = analyse(detection.warped)
        wrong = sum(
            1
            for r in range(8)
            for f in range(8)
            if bool(truth[r][f]) != reading.squares[r][f].occupied
        )
        damage.append(wrong)
        assert wrong <= 20, (
            f"seed {seed}: reported confident (score {detection.score:.2f}) but "
            f"{wrong}/64 squares had the wrong occupancy"
        )

    assert damage, "every harsh fixture declined; the threshold may be too strict"
    assert float(np.median(damage)) <= 6, (
        f"typical confident harsh reading cost {np.median(damage)} wrong squares"
    )


def test_clutter_is_not_mistaken_for_the_board():
    """Books and phones on the table are the same four-sided blob a board is."""
    image, truth = render_photo(MIDDLEGAME, Realism.typical(), seed=3)
    detection = detect_board(image)
    error = max(
        float(np.hypot(a[0] - b[0], a[1] - b[1]))
        for a, b in zip(detection.corners, truth)
    )
    assert error < 60


def test_score_still_separates_boards_from_non_boards():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (700, 700, 3), dtype=np.uint8)
    try:
        assert detect_board(noise).score < MIN_ACCEPTABLE_SCORE
    except BoardNotFound:
        pass

    image, _ = render_photo(MIDDLEGAME, Realism.typical(), seed=1)
    assert detect_board(image).score >= MIN_ACCEPTABLE_SCORE


# ---------------------------------------------------- occupancy and colour


@pytest.mark.parametrize("placement", [MIDDLEGAME, ENDGAME])
def test_occupancy_survives_shadows_and_overhang(placement):
    """Standing pieces shadow their neighbours and lean over the square behind.

    Both are ways to make an empty square look occupied, and this is the
    non-learned half of the pipeline, so it is held to a tight bound.
    """
    truth = placement_to_grid(placement)
    wrong = 0
    boards = 0
    for seed in (11, 12, 13):
        image, _ = render_photo(placement, Realism.typical(), seed=seed)
        try:
            reading = analyse(detect_board(image).warped)
        except BoardNotFound:
            continue
        boards += 1
        wrong += sum(
            1
            for r in range(8)
            for f in range(8)
            if bool(truth[r][f]) != reading.squares[r][f].occupied
        )
    assert boards, "no board was found in any fixture"
    assert wrong / boards <= 3, f"{wrong / boards:.1f} occupancy errors per board"


def test_piece_colour_survives_uneven_lighting():
    """A lighting gradient must not flip light pieces to dark across the board."""
    truth = placement_to_grid(MIDDLEGAME)
    wrong = 0
    boards = 0
    for seed in (11, 12, 13):
        image, _ = render_photo(MIDDLEGAME, Realism.typical(), seed=seed)
        try:
            reading = analyse(detect_board(image).warped)
        except BoardNotFound:
            continue
        boards += 1
        wrong += sum(
            1
            for r in range(8)
            for f in range(8)
            if truth[r][f]
            and reading.squares[r][f].occupied
            and reading.squares[r][f].is_black != truth[r][f].islower()
        )
    assert boards
    assert wrong / boards <= 3, f"{wrong / boards:.1f} colour errors per board"


# --------------------------------------------------------------- end to end


@pytest.mark.parametrize("level,floor", [("gentle", 0.80), ("typical", 0.78)])
def test_end_to_end_accuracy_on_photo_like_images(photo_classifier, level, floor):
    """The number that actually matters, on the images that actually matter.

    The floors sit below the measured 88%/86% so ordinary variation does not
    make the suite flaky, but high enough that a real regression trips them.
    They are deliberately far below the 97% the flat renderer reports.
    """
    total = correct = 0
    for placement in (MIDDLEGAME, ENDGAME):
        for seed in (11, 12):
            image, _ = render_photo(placement, getattr(Realism, level)(), seed=seed)
            try:
                grid = read_position(image, photo_classifier).grid()
            except BoardNotFound:
                continue
            truth = placement_to_grid(placement)
            for r in range(8):
                for f in range(8):
                    if truth[r][f]:
                        total += 1
                        correct += truth[r][f] == grid[r][f]
    assert total, "no board was read"
    accuracy = correct / total
    assert accuracy >= floor, f"{level}: {accuracy:.0%} of pieces correct"


def test_calibration_works_from_a_photo_like_image(tmp_path):
    classifier = PieceClassifier(tmp_path)
    image, _ = render_photo(STARTING_PLACEMENT, Realism.typical(), seed=201)
    result = calibrate_from_image(image, classifier)
    assert result.samples_added == 32
    assert set(classifier.class_counts()) == set("pnbrqk")


def test_a_second_calibration_photo_rescues_a_bad_first_one(tmp_path):
    """The advice the CLI and README give, held to account.

    Measured on these fixtures, calibration photo *quality* dominates and
    count barely matters: one good photo scores 85%, one mediocre one 81%, one
    bad one 68%, and every two-photo combination lands around 84%. So a second
    photo is not an improvement on a good first one — it is insurance against
    a bad one, which is what this pins. If it ever fails, the guidance should
    change rather than be repeated.
    """

    def accuracy(specs) -> float:
        classifier = PieceClassifier(tmp_path / f"model-{len(specs)}")
        for seed, level in specs:
            image, _ = render_photo(
                STARTING_PLACEMENT, getattr(Realism, level)(), seed=seed
            )
            calibrate_from_image(image, classifier)

        total = correct = 0
        # Spread across positions and conditions. A single position under a
        # single condition is far too small a sample to separate the two —
        # measured over this wider set the gap is 69% against 84%.
        for placement in (MIDDLEGAME, ENDGAME):
            for level in ("gentle", "typical", "harsh"):
                for seed in (11, 12):
                    image, _ = render_photo(
                        placement, getattr(Realism, level)(), seed=seed
                    )
                    try:
                        grid = read_position(image, classifier).grid()
                    except BoardNotFound:
                        continue
                    truth = placement_to_grid(placement)
                    for r in range(8):
                        for f in range(8):
                            if truth[r][f]:
                                total += 1
                                correct += truth[r][f] == grid[r][f]
        return correct / max(total, 1)

    bad_only = accuracy([(204, "harsh")])
    bad_plus_good = accuracy([(204, "harsh"), (201, "gentle")])
    assert bad_plus_good > bad_only + 0.05, (
        f"a poor calibration photo scored {bad_only:.0%} alone and "
        f"{bad_plus_good:.0%} with a good one added — if adding a second photo "
        "no longer rescues a bad first one, the advice should change"
    )
