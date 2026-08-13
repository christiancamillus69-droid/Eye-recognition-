"""Confirm the position before you trust an engine's evaluation of it.

One misread piece silently changes the evaluation, so a review step is what
makes the output trustworthy rather than merely fast. The design goal is to
make that review cost seconds, not minutes:

* Squares are queued **least-confident first**, so you check the handful the
  pipeline is unsure about instead of all thirty-two.
* Selecting a square shows the actual photograph of it, enlarged. Deciding
  "bishop or pawn?" from the real pixels beats deciding from a guess.
* Corrections are keystrokes — ``p n b r q k`` to set a type, ``x`` to empty a
  square, ``c`` to flip its colour — with no dragging or aiming.
* Every correction becomes labelled training data from your own board, so the
  queue gets shorter the more you use it.
"""

from __future__ import annotations

import io
import threading
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file

from .. import export
from ..calibrate import add_corrections
from ..classifier import PieceClassifier
from ..fen import Position, infer_castling, material_warnings, validate
from ..pipeline import read_position, recolour
from ..squares import piece_crop
from ..types import BoardReading, PIECE_TYPES

#: Squares below this confidence are always worth a second look.
FLAG_THRESHOLD = 0.6
#: The least-confident occupied squares are flagged whatever their score, so a
#: reading the model was quietly unsure about still gets seen. This does not
#: catch a *confidently* wrong prediction — nothing derived from confidence
#: can — which is why the board is drawn large next to the queue: comparing it
#: against the real board takes a chess player a couple of seconds.
ALWAYS_REVIEW = 10


def _rotate_grid_objects(grid, quarter_turns: int):
    """Rotate any 8x8 nested list clockwise, matching boardeye.fen.rotate_grid."""
    out = [row[:] for row in grid]
    for _ in range(quarter_turns % 4):
        out = [[out[7 - f][r] for f in range(8)] for r in range(8)]
    return out


@dataclass
class EditorSession:
    """Everything one review of one photograph needs."""

    image: np.ndarray
    reading: BoardReading
    classifier: PieceClassifier | None = None
    side_to_move: str = "w"
    castling: str | None = None  # None -> infer from placement
    en_passant: str = "-"
    rotation: int = 0
    learn_from_corrections: bool = True
    #: (crop, piece_type) pairs gathered from the user's edits.
    corrections: list[tuple[np.ndarray, str]] = field(default_factory=list)
    saved_to: Path | None = None

    # ------------------------------------------------------------- position

    def position(self) -> Position:
        return Position(
            grid=self.reading.grid(),
            side_to_move=self.side_to_move,
            castling=self.castling,
            en_passant=self.en_passant,
        )

    def fen(self) -> str:
        return self.position().fen()

    def rotate(self, quarter_turns: int = 1) -> None:
        """Turn the board, keeping edits and crops aligned.

        The squares and the rectified image rotate together, so the crop shown
        for a square still matches the square after a rotation. Re-running
        detection here would throw away the user's corrections.
        """
        quarter_turns %= 4
        if not quarter_turns:
            return
        self.reading.squares = _rotate_grid_objects(self.reading.squares, quarter_turns)
        if self.reading.warped is not None:
            warped = self.reading.warped
            for _ in range(quarter_turns):
                warped = np.rot90(warped, k=-1).copy()
            self.reading.warped = warped
        self.rotation = (self.rotation + quarter_turns) % 4
        # Castling rights inferred before a rotation no longer mean anything.
        self.castling = None

    def set_square(
        self, rank: int, file: int, piece_type: str | None, is_black: bool | None = None
    ) -> None:
        """Apply one user edit, and remember it as training data."""
        square = self.reading.squares[rank][file]

        if piece_type is None:  # empty the square
            square.occupied = False
            square.piece = ""
            square.piece_confidence = 1.0
            square.occupancy_confidence = 1.0
            return

        piece_type = piece_type.lower()
        if piece_type not in PIECE_TYPES:
            raise ValueError(f"not a piece type: {piece_type!r}")

        if is_black is not None:
            square.is_black = is_black
        elif square.is_black is None:
            square.is_black = False

        was_wrong = (not square.occupied) or square.piece.lower() != piece_type
        square.occupied = True
        square.piece = piece_type if square.is_black else piece_type.upper()
        # The user has now told us the answer, so this square is settled.
        square.piece_confidence = 1.0
        square.occupancy_confidence = 1.0
        square.colour_confidence = 1.0

        if was_wrong and self.reading.warped is not None:
            self.corrections.append(
                (piece_crop(self.reading.warped, rank, file), piece_type)
            )

    def recolour_square(self, rank: int, file: int) -> None:
        square = self.reading.squares[rank][file]
        if square.occupied:
            square.is_black = not bool(square.is_black)
            square.colour_confidence = 1.0
            recolour(self.reading)

    # ---------------------------------------------------------------- views

    def flagged_squares(self) -> list[tuple[int, int]]:
        """Which squares are worth a look, least trustworthy first.

        Two rules, because either alone leaks errors. An absolute confidence
        cut catches squares the model openly doubted — but a model can be
        confidently wrong, and those would never surface. So the least-certain
        handful of occupied squares are always flagged regardless of their
        absolute score. On the synthetic fixtures every misread piece fell
        inside that handful even when its own confidence looked healthy.
        """
        occupied = [
            (r, f)
            for r, f in self.reading.review_order()
            if self.reading.squares[r][f].occupied
        ]
        always = occupied[:ALWAYS_REVIEW]
        low = [
            (r, f)
            for r, f in occupied[ALWAYS_REVIEW:]
            if self.reading.squares[r][f].review_priority < FLAG_THRESHOLD
        ]
        return always + low

    def board_state(self) -> dict:
        """Everything the page needs to render, in one payload."""
        fen = self.fen()
        handoff = export.build(fen)
        flagged = set(self.flagged_squares())
        squares = []
        for r in range(8):
            for f in range(8):
                sq = self.reading.squares[r][f]
                squares.append(
                    {
                        "rank": r,
                        "file": f,
                        "piece": sq.piece,
                        "occupied": sq.occupied,
                        "is_black": bool(sq.is_black),
                        "confidence": round(sq.review_priority, 4),
                        "piece_confidence": round(sq.piece_confidence, 4),
                        "colour_confidence": round(sq.colour_confidence, 4),
                        "occupancy_confidence": round(sq.occupancy_confidence, 4),
                        "flagged": (r, f) in flagged,
                        "alternatives": {
                            k: round(v, 4) for k, v in list(sq.alternatives.items())[:4]
                        },
                    }
                )
        return {
            "squares": squares,
            "review_order": [list(p) for p in self.reading.review_order()],
            "flagged": [list(p) for p in self.flagged_squares()],
            "fen": fen,
            "lichess": handoff.lichess,
            "chesscom": handoff.chesscom,
            "warnings": validate(fen) + material_warnings(self.reading.grid()),
            "side_to_move": self.side_to_move,
            "castling": self.castling if self.castling is not None else infer_castling(
                self.reading.grid()
            ),
            "en_passant": self.en_passant,
            "detection": {
                "method": self.reading.detection_method,
                "corners": self.reading.corners,
            },
            "corrections": len(self.corrections),
            "learn": self.learn_from_corrections,
            "trained": bool(self.classifier and self.classifier.is_trained),
        }

    # ------------------------------------------------------------- learning

    def commit_corrections(self) -> str | None:
        """Fold this session's corrections into the model."""
        if not (self.learn_from_corrections and self.corrections and self.classifier):
            return None
        count = len(self.corrections)
        stats = add_corrections(self.classifier, self.corrections)
        self.corrections.clear()
        return f"Learned from {count} correction(s). Model now: {stats.summary()}"


def _png_response(image: np.ndarray):
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        return jsonify({"error": "could not encode image"}), 500
    return send_file(io.BytesIO(buffer.tobytes()), mimetype="image/png")


def create_app(session: EditorSession) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SESSION"] = session

    @app.get("/")
    def index():
        from flask import render_template

        return render_template("editor.html")

    @app.get("/api/state")
    def state():
        return jsonify(session.board_state())

    @app.get("/image/photo.png")
    def photo():
        return _png_response(session.image)

    @app.get("/image/board.png")
    def board():
        if session.reading.warped is None:
            return jsonify({"error": "no rectified board"}), 404
        return _png_response(session.reading.warped)

    @app.get("/image/crop/<int:rank>/<int:file>.png")
    def crop(rank: int, file: int):
        """The actual photograph of one square, for deciding a close call."""
        if session.reading.warped is None:
            return jsonify({"error": "no rectified board"}), 404
        if not (0 <= rank < 8 and 0 <= file < 8):
            return jsonify({"error": "square out of range"}), 404
        patch = piece_crop(session.reading.warped, rank, file)
        scaled = cv2.resize(
            patch, (patch.shape[1] * 3, patch.shape[0] * 3), interpolation=cv2.INTER_NEAREST
        )
        return _png_response(scaled)

    @app.post("/api/square")
    def set_square():
        data = request.get_json(force=True)
        rank, file = int(data["rank"]), int(data["file"])
        if not (0 <= rank < 8 and 0 <= file < 8):
            return jsonify({"error": "square out of range"}), 400
        try:
            session.set_square(
                rank, file, data.get("piece_type"), data.get("is_black")
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(session.board_state())

    @app.post("/api/recolour")
    def recolour_square():
        data = request.get_json(force=True)
        session.recolour_square(int(data["rank"]), int(data["file"]))
        return jsonify(session.board_state())

    @app.post("/api/rotate")
    def rotate():
        data = request.get_json(silent=True) or {}
        session.rotate(int(data.get("turns", 1)))
        return jsonify(session.board_state())

    @app.post("/api/settings")
    def settings():
        data = request.get_json(force=True)
        if "side_to_move" in data:
            side = data["side_to_move"]
            if side not in ("w", "b"):
                return jsonify({"error": "side_to_move must be 'w' or 'b'"}), 400
            session.side_to_move = side
        if "castling" in data:
            value = (data["castling"] or "").strip()
            session.castling = value if value else "-"
        if "en_passant" in data:
            session.en_passant = (data["en_passant"] or "-").strip() or "-"
        if "learn" in data:
            session.learn_from_corrections = bool(data["learn"])
        return jsonify(session.board_state())

    @app.post("/api/corners")
    def corners():
        """Re-run the pipeline from four corners the user clicked on the photo."""
        data = request.get_json(force=True)
        points = data.get("corners")
        if not points or len(points) != 4:
            return jsonify({"error": "need exactly four corners"}), 400
        try:
            session.reading = read_position(
                session.image, session.classifier, corners=points
            )
        except Exception as exc:  # detection can fail on nonsense corners
            return jsonify({"error": f"could not use those corners: {exc}"}), 400
        session.rotation = 0
        session.castling = None
        session.corrections.clear()
        return jsonify(session.board_state())

    @app.post("/api/commit")
    def commit():
        message = session.commit_corrections()
        state = session.board_state()
        state["message"] = message
        return jsonify(state)

    @app.get("/download/position.pgn")
    def pgn():
        data = export.to_pgn(session.fen()).encode("utf-8")
        return send_file(
            io.BytesIO(data),
            mimetype="application/x-chess-pgn",
            as_attachment=True,
            download_name="position.pgn",
        )

    return app


def serve(
    session: EditorSession,
    host: str = "127.0.0.1",
    port: int = 5000,
    open_browser: bool = True,
) -> None:
    """Run the editor until the user stops it."""
    app = create_app(session)
    url = f"http://{host}:{port}/"
    if open_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    print(f"BoardEye editor running at {url}  (Ctrl-C to stop)")
    app.run(host=host, port=port, debug=False, use_reloader=False)
