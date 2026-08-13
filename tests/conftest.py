"""Shared fixtures.

The trained classifier is session-scoped because calibrating it costs a couple
of seconds and every recognition test wants the same one.
"""

from __future__ import annotations

import pytest

from boardeye.calibrate import calibrate_from_image
from boardeye.classifier import PieceClassifier

from .render import STARTING_FEN, render_photo

#: Positions used across the recognition tests, with their ground-truth
#: placement fields.
POSITIONS = {
    "middlegame": "r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1B3/PPP2PPP/R2Q1RK1",
    "tactical": "r4rk1/1bq2ppp/p2p1n2/1p2p3/3NP3/1BN5/PPP2PPP/R2Q1RK1",
    "endgame": "8/5pk1/6p1/8/1P6/2K5/5PP1/8",
    "sparse": "8/8/4k3/8/8/2K5/8/8",
}

#: Two calibration photos under different conditions. Measured to hold ~97%
#: piece accuracy across varied scan conditions where one photo drops to 78%.
CALIBRATION_CONDITIONS = [
    dict(tilt=0.08, blur=0.3, noise=0.005),
    dict(tilt=0.22, blur=1.0, noise=0.020),
]


@pytest.fixture(scope="session")
def trained_classifier(tmp_path_factory) -> PieceClassifier:
    model_dir = tmp_path_factory.mktemp("model")
    classifier = PieceClassifier(model_dir)
    for index, condition in enumerate(CALIBRATION_CONDITIONS):
        photo, _ = render_photo(STARTING_FEN, seed=11 + index, **condition)
        calibrate_from_image(photo, classifier)
    return classifier


@pytest.fixture
def scan_photo():
    """Render a photograph of a position under typical scan conditions."""

    def _make(placement: str, **overrides):
        options = dict(tilt=0.15, blur=0.8, noise=0.015, seed=3)
        options.update(overrides)
        photo, corners = render_photo(placement, **options)
        return photo, corners

    return _make
