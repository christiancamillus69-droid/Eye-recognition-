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

**Make it a good photo.** This is the single highest-leverage thing you can do:
sharp, evenly lit, board filling the frame. Measured on photograph-like test
images, a good calibration photo gave 85% piece accuracy and a poor one 68%,
and no amount of extra photos closed that gap.

Taking a second one is cheap insurance rather than an improvement — it barely
helps a good first photo, but it pulls a bad first photo back up to about 84%:

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
boardeye check start-1.jpg start-2.jpg   # grade photos, measure expected accuracy
boardeye scan --camera 0        # read from a webcam instead of a file
boardeye scan photo.jpg --open lichess   # skip review, go straight to the site
boardeye status                 # what the model has learned so far
boardeye retrain                # refit from stored crops
boardeye scan photo.jpg --debug out/     # dump every intermediate image
```

---

## Taking a good photo

Don't take my word for whether a photo is good enough — ask:

```bash
boardeye check start-1.jpg start-2.jpg
```

It grades each photo and names what to change. Given two or more photos of the
starting position it does something better: it trains on all but one, reads the
one left out, and since the answer is known, reports **the accuracy you can
expect on your own board**, along with which piece types it struggles with.

```
start-1.jpg  [good]
  detection 0.78 | sharpness 1161 | board fills 63% of frame | clipped 0.6%
  nothing to improve

  78% of pieces read correctly (64 pieces over 2 held-out photos)
  weakest piece types — b: 5/8, n: 5/8, q: 1/4
```

That number is worth more than anything in this README, because it is measured
on your set, your lighting and your camera rather than on my test fixtures.

What actually matters, in order:

1. **Shoot from as far overhead as you can.** This is the biggest single
   lever. It removes perspective distortion, stops tall pieces from hiding the
   ones behind them, and makes a piece look the same wherever it stands.
   Angled shots do work — calibration learns whatever viewpoint you use — but
   overhead is where the tool is strongest.
2. **Get the board filling most of the frame.** Below about 15% of the frame,
   there are too few pixels per piece to tell a bishop from a pawn.
3. **Focus and light.** Blur is the most common killer, and it usually comes
   from a dim room: the phone lengthens the exposure and a handheld shot
   smears. Turn a light on rather than holding still harder.
4. **Even light, no glare.** A hard shadow across half the board, or a lamp
   glaring off polished pieces, costs more than a slightly awkward angle.
5. **Keep straight-edged clutter out of frame.** A book or a phone beside the
   board is the same four-sided shape the detector is hunting for.

Then scan under roughly the lighting you calibrated in. If scans start coming
out poorly, re-running `boardeye calibrate` under the current conditions
usually fixes it faster than fighting individual positions.

---

## What accuracy to expect

The test suite generates boards two ways, and the difference between them is
the most useful thing in this section.

The first draws flat 2-D piece symbols on flat colour under even light. The
second renders pieces that stand up and lean over the square behind them, cast
shadows onto their neighbours, on wood grain, under a lighting gradient, with
clutter on the table, camera roll and JPEG artefacts.

| | flat diagrams | photograph-like |
| --- | --- | --- |
| Pieces identified correctly | 97.7% | **86%** |
| Corrections needed per board | ~1 | **~4** |
| Occupancy errors | none | ~2 per board |
| Piece colour errors | none | ~1 per board |

**The flat number overstated real performance by about ten points, and the
honest figure is the right-hand column.** In good conditions it reaches 88%,
in poor ones 76%. Almost all the loss is in piece *type*: where a piece is and
what colour it is stay reliable, because those come from geometry rather than
from a learned model.

Even the right-hand column is not a photograph of your board, and your set,
lighting and camera will differ. What it is good for is the shape of the
problem: expect to fix a few pieces per position, which is why the review
screen is built for fast correction rather than for being trusted blindly.

Board detection, on the same photograph-like images:

| conditions | board found and trusted |
| --- | --- |
| good | 10 of 10 |
| typical | 10 of 10 |
| poor | 7 of 10 |

The three failures are the system declining rather than guessing — it scores
its own detection and refuses below a threshold, which sends you to the
"click the four corners" fallback instead of handing you a scrambled board.

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

130 tests, about 90 seconds. They cover FEN handling, board detection,
occupancy and colour, the classifier and calibration, and the editor's server
side. Board images are generated from known positions, so everything has
ground truth without needing photographs.

There are two generators, and the split matters:

- `tests/render.py` draws flat piece symbols on flat colour. Fast, and fine
  for geometry and plumbing.
- `tests/photoreal.py` renders standing pieces that overhang the square
  behind, cast shadows, on wood grain under uneven light with table clutter,
  camera roll and JPEG artefacts. Slower, and the only one whose numbers are
  worth quoting — the flat generator flattered the pipeline by ten points, and
  `tests/test_photoreal.py` holds the thresholds that survived the harder one.

Layout:

| file | what it does |
| --- | --- |
| `detect.py` | finds the board, corrects perspective, scores its own confidence |
| `squares.py` | cuts the board into squares; occupancy and piece colour |
| `classifier.py` | the learned piece-type model and its features |
| `calibrate.py` | turns a starting-position photo into labelled training data |
| `pipeline.py` | joins the three stages together |
| `quality.py` | grades photos; held-out accuracy on a real board |
| `fen.py` | FEN assembly, castling inference, position and material checks |
| `export.py` | FEN, Lichess and Chess.com URLs, PGN |
| `editor/` | the local review page |
| `camera.py` | live webcam scanning |

Your trained model and its crops live in `~/.boardeye/`. Delete that directory
to start over.

### Real photographs are the missing piece

Every number quoted here comes from generated images. The pipeline has never
been run against a photograph of an actual wooden set, because it was built
somewhere without a camera or access to one.

If you add photos of your board to `samples/` — two of the starting position,
plus a few real positions with their correct FEN alongside — that gap closes.
`samples/README.md` explains the layout and how to write the FEN files without
typing them by hand. With a photo and its right answer, an error stops being
an impression and becomes something countable, and therefore fixable.

---

## Not included

No move detection across frames, no clock or scoresheet reading, and no deep
learning — the scikit-learn model is sized for the amount of data one
calibration photo provides, and a neural network would need far more. If your
saved corrections ever grow into a few thousand crops, that trade changes, and
the recogniser sits behind a small interface so it can be swapped without
touching the rest.
