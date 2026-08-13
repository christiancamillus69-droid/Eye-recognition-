"""Hand a position off to an engine.

Three routes, because the sites accept different things and people work
differently: a bare FEN to paste, a URL that opens the position directly, and
a PGN file that imports into either site.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from urllib.parse import quote

LICHESS_ANALYSIS = "https://lichess.org/analysis"
CHESSCOM_ANALYSIS = "https://www.chess.com/analysis"


def lichess_url(fen: str) -> str:
    """Open a position on the Lichess analysis board.

    Lichess takes the FEN in the path with spaces written as underscores; the
    slashes separating ranks stay as they are.
    """
    return f"{LICHESS_ANALYSIS}/{fen.strip().replace(' ', '_')}"


def chesscom_url(fen: str) -> str:
    """Open a position on the Chess.com analysis board."""
    return f"{CHESSCOM_ANALYSIS}?fen={quote(fen.strip(), safe='')}&tab=analysis"


def to_pgn(fen: str, event: str = "BoardEye scan", when: date | None = None) -> str:
    """A minimal PGN carrying the position, importable by both sites.

    ``SetUp`` and ``FEN`` are the tags that make a PGN describe a position
    rather than a game from the initial array; without ``SetUp "1"`` some
    readers ignore the FEN tag entirely.
    """
    when = when or date.today()
    tags = [
        ("Event", event),
        ("Site", "?"),
        ("Date", when.strftime("%Y.%m.%d")),
        ("Round", "?"),
        ("White", "?"),
        ("Black", "?"),
        ("Result", "*"),
        ("SetUp", "1"),
        ("FEN", fen.strip()),
    ]
    header = "\n".join(f'[{name} "{value}"]' for name, value in tags)
    return f"{header}\n\n*\n"


@dataclass
class Handoff:
    """Everything needed to get this position in front of an engine."""

    fen: str
    lichess: str
    chesscom: str
    pgn: str

    def as_dict(self) -> dict[str, str]:
        return {
            "fen": self.fen,
            "lichess": self.lichess,
            "chesscom": self.chesscom,
            "pgn": self.pgn,
        }


def build(fen: str, event: str = "BoardEye scan") -> Handoff:
    """Produce every handoff format for one FEN."""
    fen = fen.strip()
    return Handoff(
        fen=fen,
        lichess=lichess_url(fen),
        chesscom=chesscom_url(fen),
        pgn=to_pgn(fen, event=event),
    )
