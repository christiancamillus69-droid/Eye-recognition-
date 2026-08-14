"""BoardEye — photograph an over-the-board chess position, get an engine-ready FEN.

Everything here runs locally: board geometry comes from classical computer
vision, and piece type comes from a small classifier trained on photographs of
your own board. No API key, no network calls at runtime.
"""

__version__ = "0.1.0"
