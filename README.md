# BoardEye

Photograph a chess position on a real board and get a FEN, plus a link that
opens it straight in the Lichess or Chess.com analysis board.

The problem it solves is the tedious one: after an over-the-board game you want
an engine's opinion, and setting the position up by hand means placing thirty
pieces one at a time. BoardEye reads the board from a photo, shows you what it
read so you can fix anything wrong, and hands you the link.

**It runs entirely on your machine.** No API key, no account, no network calls,
no cost. It works offline.

---

## How it works, in one paragraph

Classical computer vision finds the board in the photo, corrects the
perspective, and works out which squares hold a piece and whether that piece is
light or dark. That part is geometry and needs no training. Working out
*which* piece — bishop or pawn — is the hard part, and it is handled by a small
classifier you train yourself in about ten seconds. You photograph your board
set up for a new game; because the starting position is known, every piece
labels itself, so one photo yields 32 labelled examples across all six piece
types with nothing to annotate. Afterwards, every correction you make in the
review screen is added to that training set, so it gets better at *your* pieces
the more you use it.

---

## Setup

```bash
git clone <this repo> && cd Eye-recognition-
python3 -m venv .venv
.venv/bin/pip install -e .
```

Then teach it your pieces:

```bash
boardeye calibrate starting-position.jpg
```

That is a photo of your own board with all 32 pieces on their home squares.
**Take two, in different light and from different angles** — measured on the
test fixtures, two photos held 97% accuracy across varied conditions where one
photo fell to 78% when the lighting changed:

```bash
boardeye calibrate start-1.jpg start-2.jpg
```

It does not matter which side of the board you shoot from, or whether you hold
the phone in portrait. Calibration works out the orientation itself and rotates
to match.

## Using it

```bash
boardeye scan position.jpg
```

This opens a page in your browser with the position it read. Check it, fix
anything wrong, click **Open in Lichess**. Fixing is keyboard-driven — the
squares it was least sure about are queued up, `Space` steps through them, and
`p n b r q k` sets a piece:

| key | action |
| --- | --- |
| `p` `n` `b` `r` `q` `k` | set the selected square to that piece |
| `x` | empty the square |
| `c` | flip the piece's colour |
| `Space` | jump to the next uncertain square |
| arrow keys | move around the board |

If you just want the FEN in your terminal:

```bash
boardeye scan position.jpg --fen-only
```

Other commands:

```bash
boardeye scan --camera 0        # read from a webcam instead of a file
boardeye scan photo.jpg --open lichess   # skip review, go straight to the site
boardeye status                 # what the model has learned so far
boardeye retrain                # refit from stored crops
boardeye scan photo.jpg --debug out/     # dump every intermediate image
```

---

## Taking a good photo

**Shoot from directly above if you can.** A near-overhead phone photo is the
single biggest thing you can do for accuracy. It removes perspective
distortion, stops tall pieces from hiding the ones behind them, and makes a
piece look the same wherever it stands on the board. Angled shots work too —
calibration learns whatever viewpoint you use — but overhead works best.

Beyond that: get the board to fill most of the frame, avoid a hard shadow
falling across part of it, and try to scan under roughly the same lighting you
calibrated in. If a scan comes out unusually bad, running `boardeye calibrate`
once more under the current lighting usually fixes it.

---

## What accuracy to expect

Measured on synthetic test boards across six positions and four photographic
conditions, after calibrating from two photos:

| | |
| --- | --- |
| Pieces identified correctly | **97.7%** |
| Boards needing no correction at all | 14 of 24 |
| Errors appearing in the ten flagged squares | 10 of 12 |
| Occupancy and piece colour | no errors |

**These numbers come from rendered images, not photographs of a real board.**
They show the pipeline works and give a sense of the shape of its errors. They
are not a prediction of what you will get on a wooden set under a kitchen
light, which will be worse. The only way to find out is to try it, which is why
the review screen exists and why it is built for fast correction rather than
for being trusted blindly.

Some things worth knowing about how it fails:

- **Occupancy and colour are the reliable parts.** Where a piece is, and
  whether it is yours or your opponent's, come out of geometry and brightness
  rather than a learned model. Errors concentrate in piece *type*.
- **Queens and kings are the weakest.** A calibration photo contains sixteen
  pawns but only two queens and two kings, so those classes have the least to
  learn from. Correcting them in the editor is the fastest way to improve it.
- **The flags are a priority list, not a guarantee.** Most mistakes are made
  with low confidence and get flagged. Some are not: the model is occasionally
  confident and wrong, and nothing derived from confidence can catch that. This
  is why the board is drawn large next to the queue — a chess player spots a
  bishop standing where a pawn should be in about a second.
- **A near-empty board is hard.** With two pieces there is no context, and a
  lone king is readily read as a queen. Both pieces get flagged, so it costs
  two keystrokes.
- **Impossible material is caught automatically.** If a misread leaves you with
  nine pawns or three bishops that no promotion could explain, the page says
  so before you export.

---

## The live scanner

```bash
boardeye scan --camera 0
```

Watches the camera and captures by itself once the board has been detected in
the same place for several frames running, which filters out motion blur and
the moment while autofocus settles.

**This is the one part that has never been run.** It was developed in a
container with no camera attached, so while the geometry it relies on is
tested, the capture loop around it is not. If it misbehaves, `boardeye scan`
on a still photo does the same job.

The default install uses `opencv-python-headless`, which has no window support,
so the scanner captures automatically without a live preview. For a preview
showing the detected board:

```bash
.venv/bin/pip uninstall -y opencv-python-headless && .venv/bin/pip install opencv-python
```

---

## Development

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

118 tests, about 30 seconds. They cover FEN handling, board detection,
occupancy and colour, the classifier and calibration, and the editor's server
side. Board images are generated from known positions by `tests/render.py`, so
everything has ground truth without needing photographs.

Layout:

| file | what it does |
| --- | --- |
| `detect.py` | finds the board, corrects perspective, scores its own confidence |
| `squares.py` | cuts the board into squares; occupancy and piece colour |
| `classifier.py` | the learned piece-type model and its features |
| `calibrate.py` | turns a starting-position photo into labelled training data |
| `pipeline.py` | joins the three stages together |
| `fen.py` | FEN assembly, castling inference, position and material checks |
| `export.py` | FEN, Lichess and Chess.com URLs, PGN |
| `editor/` | the local review page |
| `camera.py` | live webcam scanning |

Your trained model and its crops live in `~/.boardeye/`. Delete that directory
to start over.

---

## Not included

No move detection across frames, no clock or scoresheet reading, and no deep
learning — the scikit-learn model is sized for the amount of data one
calibration photo provides, and a neural network would need far more. If your
saved corrections ever grow into a few thousand crops, that trade changes, and
the recogniser sits behind a small interface so it can be swapped without
touching the rest.
