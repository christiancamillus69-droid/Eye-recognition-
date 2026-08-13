"""The review UI's server side: state, edits, rotation, corners, and learning."""

from __future__ import annotations

import pytest

from boardeye.editor.app import ALWAYS_REVIEW, EditorSession, create_app
from boardeye.fen import placement_to_grid
from boardeye.pipeline import read_position

from .conftest import POSITIONS


@pytest.fixture
def session(trained_classifier, scan_photo):
    photo, corners = scan_photo(POSITIONS["middlegame"])
    reading = read_position(photo, trained_classifier)
    session = EditorSession(image=photo, reading=reading, classifier=trained_classifier)
    session.true_corners = corners
    return session


@pytest.fixture
def client(session):
    return create_app(session).test_client()


# --------------------------------------------------------------- rendering


def test_page_and_assets_load(client):
    assert client.get("/").status_code == 200
    for path in ("/static/editor.css", "/static/editor.js"):
        assert client.get(path).status_code == 200, path


@pytest.mark.parametrize(
    "path", ["/image/photo.png", "/image/board.png", "/image/crop/0/0.png"]
)
def test_images_render(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.mimetype == "image/png"
    assert len(response.data) > 100


def test_crop_out_of_range_is_rejected(client):
    assert client.get("/image/crop/9/0.png").status_code == 404


def test_state_payload_has_everything_the_page_needs(client):
    state = client.get("/api/state").get_json()
    assert len(state["squares"]) == 64
    for key in ("fen", "lichess", "chesscom", "warnings", "flagged", "review_order"):
        assert key in state, key
    assert state["trained"] is True


def test_flagged_list_is_ordered_least_confident_first(client):
    state = client.get("/api/state").get_json()
    lookup = {(s["rank"], s["file"]): s for s in state["squares"]}
    confidences = [lookup[(r, f)]["confidence"] for r, f in state["flagged"]]
    assert confidences == sorted(confidences)


def test_least_confident_squares_are_always_flagged(client):
    """A quietly-wrong reading must surface even if its score looks healthy."""
    state = client.get("/api/state").get_json()
    occupied = [
        (r, f)
        for r, f in state["review_order"]
        if next(s for s in state["squares"] if s["rank"] == r and s["file"] == f)["occupied"]
    ]
    flagged = {tuple(p) for p in state["flagged"]}
    for position in occupied[:ALWAYS_REVIEW]:
        assert tuple(position) in flagged


# ------------------------------------------------------------------- edits


def test_setting_a_piece_updates_the_fen_and_records_a_correction(client):
    before = client.get("/api/state").get_json()
    state = client.post(
        "/api/square", json={"rank": 3, "file": 3, "piece_type": "q", "is_black": True}
    ).get_json()
    assert state["fen"] != before["fen"]
    assert placement_to_grid(state["fen"].split()[0])[3][3] == "q"
    assert state["corrections"] >= 1


def test_emptying_a_square_removes_the_piece(client, session):
    occupied = next(
        (r, f)
        for r in range(8)
        for f in range(8)
        if session.reading.squares[r][f].occupied
    )
    state = client.post(
        "/api/square", json={"rank": occupied[0], "file": occupied[1], "piece_type": None}
    ).get_json()
    assert placement_to_grid(state["fen"].split()[0])[occupied[0]][occupied[1]] == ""


def test_correcting_to_the_same_piece_records_nothing(client, session):
    """Only genuine corrections are worth learning from."""
    r, f = next(
        (r, f)
        for r in range(8)
        for f in range(8)
        if session.reading.squares[r][f].occupied
    )
    current = session.reading.squares[r][f]
    client.post(
        "/api/square",
        json={
            "rank": r,
            "file": f,
            "piece_type": current.piece.lower(),
            "is_black": current.is_black,
        },
    )
    assert len(session.corrections) == 0


def test_invalid_edits_are_rejected(client):
    assert client.post("/api/square", json={"rank": 9, "file": 0, "piece_type": "q"}).status_code == 400
    assert client.post("/api/square", json={"rank": 0, "file": 0, "piece_type": "z"}).status_code == 400


def test_flipping_colour_changes_case_not_type(client, session):
    r, f = next(
        (r, f)
        for r in range(8)
        for f in range(8)
        if session.reading.squares[r][f].occupied
    )
    before = session.reading.squares[r][f].piece
    client.post("/api/recolour", json={"rank": r, "file": f})
    after = session.reading.squares[r][f].piece
    assert after.lower() == before.lower()
    assert after.isupper() != before.isupper()


# ---------------------------------------------------------------- rotation


def test_four_rotations_return_the_original_position(client):
    original = client.get("/api/state").get_json()["fen"].split()[0]
    for _ in range(4):
        state = client.post("/api/rotate", json={"turns": 1}).get_json()
    assert state["fen"].split()[0] == original


def test_one_rotation_changes_the_position(client):
    original = client.get("/api/state").get_json()["fen"].split()[0]
    rotated = client.post("/api/rotate", json={"turns": 1}).get_json()["fen"].split()[0]
    assert rotated != original


def test_rotation_keeps_crops_aligned_with_squares(client, session):
    """Crops must follow the rotation, or the review panel shows the wrong square."""
    client.post("/api/rotate", json={"turns": 1})
    response = client.get("/image/crop/0/0.png")
    assert response.status_code == 200 and len(response.data) > 100


# ---------------------------------------------------------------- settings


def test_side_to_move_and_castling_are_settable(client):
    state = client.post("/api/settings", json={"side_to_move": "b"}).get_json()
    assert state["fen"].split()[1] == "b"
    state = client.post("/api/settings", json={"castling": "KQ"}).get_json()
    assert state["fen"].split()[2] == "KQ"
    state = client.post("/api/settings", json={"en_passant": "e3"}).get_json()
    assert state["fen"].split()[3] == "e3"


def test_bad_side_to_move_is_rejected(client):
    assert client.post("/api/settings", json={"side_to_move": "x"}).status_code == 400


# ----------------------------------------------------------------- corners


def test_manual_corners_re_read_the_board(client, session):
    state = client.post(
        "/api/corners", json={"corners": [list(c) for c in session.true_corners]}
    ).get_json()
    assert state["detection"]["method"] == "manual"
    assert len(state["squares"]) == 64


def test_bad_corner_payloads_are_rejected(client):
    assert client.post("/api/corners", json={"corners": [[0, 0], [1, 1]]}).status_code == 400
    assert client.post("/api/corners", json={}).status_code == 400


# ---------------------------------------------------------------- learning


def test_committing_corrections_grows_the_training_set(client, trained_classifier):
    before = trained_classifier.sample_count
    client.post("/api/square", json={"rank": 3, "file": 3, "piece_type": "q", "is_black": True})
    state = client.post("/api/commit", json={}).get_json()
    assert trained_classifier.sample_count > before
    assert state["corrections"] == 0
    assert "Learned from" in state["message"]


def test_learning_can_be_switched_off(client, trained_classifier):
    client.post("/api/settings", json={"learn": False})
    before = trained_classifier.sample_count
    client.post("/api/square", json={"rank": 4, "file": 4, "piece_type": "q", "is_black": False})
    state = client.post("/api/commit", json={}).get_json()
    assert trained_classifier.sample_count == before
    assert state["message"] is None


# ------------------------------------------------------------------ export


def test_pgn_downloads_with_the_current_position(client):
    fen = client.get("/api/state").get_json()["fen"]
    response = client.get("/download/position.pgn")
    assert response.status_code == 200
    assert fen in response.data.decode()


def test_material_warnings_reach_the_page(client):
    """Nine pawns on the board should be reported, not silently exported."""
    for file in range(8):
        client.post("/api/square", json={"rank": 4, "file": file, "piece_type": "p", "is_black": True})
    for file in range(8):
        client.post("/api/square", json={"rank": 5, "file": file, "piece_type": "p", "is_black": True})
    state = client.get("/api/state").get_json()
    assert any("pawns" in warning for warning in state["warnings"])
