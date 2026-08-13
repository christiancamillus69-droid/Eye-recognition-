"""Occupancy and piece colour — the part of the pipeline that needs no model."""

from __future__ import annotations

import pytest

from boardeye.detect import detect_board
from boardeye.fen import placement_to_grid
from boardeye.squares import analyse, cell, piece_crop
from boardeye.types import square_name

from .conftest import POSITIONS
from .render import render_photo


def read(placement: str, **kwargs):
    photo, _ = render_photo(placement, blur=0.8, noise=0.015, seed=3, **kwargs)
    return analyse(detect_board(photo).warped)


@pytest.mark.parametrize("name,placement", sorted(POSITIONS.items()))
@pytest.mark.parametrize("tilt", [0.0, 0.20])
def test_occupancy_is_exact(name, placement, tilt):
    reading = read(placement, tilt=tilt)
    truth = placement_to_grid(placement)
    wrong = [
        square_name(r, f)
        for r in range(8)
        for f in range(8)
        if bool(truth[r][f]) != reading.squares[r][f].occupied
    ]
    assert not wrong, f"{name}: occupancy wrong on {wrong}"


@pytest.mark.parametrize("name,placement", sorted(POSITIONS.items()))
@pytest.mark.parametrize("tilt", [0.0, 0.20])
def test_piece_colour_is_exact(name, placement, tilt):
    reading = read(placement, tilt=tilt)
    truth = placement_to_grid(placement)
    wrong = [
        square_name(r, f)
        for r in range(8)
        for f in range(8)
        if truth[r][f] and reading.squares[r][f].is_black != truth[r][f].islower()
    ]
    assert not wrong, f"{name}: colour wrong on {wrong}"


def test_empty_squares_are_confident():
    reading = read(POSITIONS["middlegame"], tilt=0.1)
    empties = [
        reading.squares[r][f]
        for r in range(8)
        for f in range(8)
        if not reading.squares[r][f].occupied
    ]
    assert empties
    assert all(square.occupancy_confidence > 0.5 for square in empties)


def test_piece_crop_includes_headroom_above_the_square():
    """A 3-D piece leans above its own square; cropping tight decapitates it."""
    reading = read(POSITIONS["middlegame"], tilt=0.15)
    warped = reading.warped
    assert piece_crop(warped, 4, 4).shape[0] > cell(warped, 4, 4).shape[0]


def test_crop_at_the_top_edge_stays_in_bounds():
    reading = read(POSITIONS["middlegame"], tilt=0.15)
    crop = piece_crop(reading.warped, 0, 0)
    assert crop.shape[0] > 0 and crop.shape[1] > 0


def test_single_colour_board_reports_no_colour_confidence():
    """With only one side's pieces present, the light/dark split is meaningless."""
    reading = read("8/8/8/3k4/4k3/8/8/8", tilt=0.0)
    occupied = [s for rank in reading.squares for s in rank if s.occupied]
    assert occupied
    assert all(square.colour_confidence == 0.0 for square in occupied)


def test_review_order_puts_least_confident_first():
    reading = read(POSITIONS["middlegame"], tilt=0.15)
    order = reading.review_order()
    priorities = [reading.squares[r][f].review_priority for r, f in order]
    assert priorities == sorted(priorities)
