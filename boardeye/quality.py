"""Tell the user whether a photo is good enough, before they rely on it.

Advice in a README is guesswork applied to someone else's board. This measures
the photo in front of it and says what to change.

The most useful thing here is not any single metric — measured against
accuracy, the individual properties of a *scan* photo predict surprisingly
little, because which pieces happen to be on the board matters more. What does
predict a lot is the quality of the **calibration** photo, and that can be
measured directly rather than inferred: given two or more photos of the
starting position, train on all but one and read the one left out. Since the
starting position is known, that yields a true held-out accuracy figure for
that particular board, pieces, camera and lighting. No proxy required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibrate import STARTING_GRID, calibrate_from_image
from .classifier import PieceClassifier
from .detect import MIN_ACCEPTABLE_SCORE, BoardNotFound, detect_board
from .pipeline import read_position
from .squares import analyse

# Thresholds are medians measured across the photo-like test fixtures at three
# quality levels (see tests/photoreal.py). "Good" sits near the gentle level,
# "poor" near the harsh one.
SHARPNESS_POOR = 350.0     # harsh fixtures sat at ~310
SHARPNESS_GOOD = 650.0     # typical fixtures at ~634, gentle at ~1067
FILL_POOR = 0.15           # board occupying under a sixth of the frame
FILL_GOOD = 0.30
CLIPPING_POOR = 0.06       # fraction of pixels crushed to black or blown out
SCORE_GOOD = 0.75          # above this a reading averaged 1.5 wrong squares


@dataclass
class PhotoReport:
    """What one photograph looks like to the pipeline."""

    label: str
    found_board: bool = False
    score: float = 0.0
    sharpness: float = 0.0
    fill: float = 0.0
    clipping: float = 0.0
    looks_like_start: bool | None = None
    occupancy_mismatch: int | None = None
    problems: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.found_board and not self.problems


def inspect(image: np.ndarray, label: str = "photo", *, expect_start: bool = False) -> PhotoReport:
    """Measure one photograph and say what would need to change."""
    report = PhotoReport(label=label)

    try:
        detection = detect_board(image)
    except BoardNotFound:
        report.problems.append("No board found in this photo at all.")
        report.advice.append(
            "Fill more of the frame with the board, and make sure all four edges "
            "are visible."
        )
        return report

    report.found_board = True
    report.score = detection.score

    grey_board = cv2.cvtColor(detection.warped, cv2.COLOR_BGR2GRAY)
    report.sharpness = float(cv2.Laplacian(grey_board, cv2.CV_64F).var())

    quad = np.array(detection.corners, dtype=np.float32)
    frame_area = image.shape[0] * image.shape[1]
    report.fill = float(cv2.contourArea(quad)) / frame_area

    grey_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    report.clipping = float(((grey_frame < 6) | (grey_frame > 249)).mean())

    if detection.score < MIN_ACCEPTABLE_SCORE:
        report.problems.append(
            f"The board was found but only scored {detection.score:.2f}, below the "
            f"{MIN_ACCEPTABLE_SCORE:.2f} needed to trust it."
        )
        report.advice.append(
            "Shoot straighter down over the board, and keep anything with strong "
            "straight edges (books, phones, placemats) out of the frame."
        )
    elif detection.score < SCORE_GOOD:
        report.advice.append(
            f"Board detection is workable but not strong ({detection.score:.2f}). "
            "A more overhead angle usually lifts it."
        )

    if report.sharpness < SHARPNESS_POOR:
        report.problems.append(f"The board is soft or out of focus ({report.sharpness:.0f}).")
        report.advice.append(
            "Steady the camera, tap to focus on the board, and add light — phones "
            "lengthen the exposure in dim rooms, which is what blurs handheld shots."
        )
    elif report.sharpness < SHARPNESS_GOOD:
        report.advice.append(
            f"Slightly soft ({report.sharpness:.0f}). Usable, but sharper would help."
        )

    if report.fill < FILL_POOR:
        report.problems.append(
            f"The board fills only {report.fill:.0%} of the frame."
        )
        report.advice.append("Get closer, or crop to the board before scanning.")
    elif report.fill < FILL_GOOD:
        report.advice.append(
            f"The board fills {report.fill:.0%} of the frame; closer would be better."
        )

    if report.clipping > CLIPPING_POOR:
        report.problems.append(
            f"{report.clipping:.0%} of the photo is pure black or pure white."
        )
        report.advice.append(
            "Avoid direct sun or a lamp glaring off the pieces; even, indirect "
            "light works best."
        )

    if expect_start:
        reading = analyse(detection.warped)
        mismatch = _starting_position_mismatch(reading)
        report.occupancy_mismatch = mismatch
        report.looks_like_start = mismatch <= 2
        if not report.looks_like_start:
            report.problems.append(
                f"This does not look like the starting position — {mismatch} squares "
                "differ from the expected layout."
            )
            report.advice.append(
                "Calibration needs all 32 pieces on their home squares."
            )

    return report


def _starting_position_mismatch(reading) -> int:
    """Fewest squares differing from the start, over all four rotations."""
    best = 64
    squares = reading.squares
    for _ in range(4):
        mismatch = sum(
            1
            for r in range(8)
            for f in range(8)
            if squares[r][f].occupied != bool(STARTING_GRID[r][f])
        )
        best = min(best, mismatch)
        squares = [[squares[7 - f][r] for f in range(8)] for r in range(8)]
    return best


@dataclass
class HoldOutResult:
    """A real accuracy figure for this board, measured not guessed."""

    accuracy: float
    pieces: int
    folds: int
    per_type: dict[str, tuple[int, int]] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.accuracy:.0%} of pieces read correctly "
            f"({self.pieces} pieces over {self.folds} held-out photos)"
        )


def holdout_accuracy(
    images: list[np.ndarray], model_dir=None
) -> HoldOutResult | None:
    """Train on all but one starting-position photo, read the one left out.

    Requires at least two photos. Because the answer is known in advance, this
    reports how well the tool will actually do on this board rather than how
    well it did on someone else's rendered fixtures.
    """
    if len(images) < 2:
        return None

    import tempfile

    correct = total = 0
    folds = 0
    per_type: dict[str, list[int]] = {}

    for held_out in range(len(images)):
        classifier = PieceClassifier(model_dir or tempfile.mkdtemp())
        trained_on = 0
        for index, image in enumerate(images):
            if index == held_out:
                continue
            try:
                calibrate_from_image(image, classifier, train=False)
                trained_on += 1
            except Exception:
                continue
        if not trained_on:
            continue
        try:
            classifier.train()
            grid = read_position(images[held_out], classifier).grid()
        except Exception:
            continue

        folds += 1
        for r in range(8):
            for f in range(8):
                expected = STARTING_GRID[r][f]
                if not expected:
                    continue
                total += 1
                hit = grid[r][f] == expected
                correct += hit
                bucket = per_type.setdefault(expected.lower(), [0, 0])
                bucket[0] += hit
                bucket[1] += 1

    if not total:
        return None
    return HoldOutResult(
        accuracy=correct / total,
        pieces=total,
        folds=folds,
        per_type={k: (v[0], v[1]) for k, v in sorted(per_type.items())},
    )
