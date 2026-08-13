"""Decide which *type* of piece stands on a square.

Three decisions shape this module, and they are all consequences of having
almost no training data:

**It classifies type only, not colour.** Six classes instead of twelve.
Piece colour already came out of :mod:`boardeye.squares` by clustering
brightness, which is reliable, so asking a learned model to redo it would
double the classes for nothing.

**Features describe shape, not colour.** Every crop is reduced to a
"piece-ness" map — how far each pixel is from the background colour measured at
that crop's own border — so the absolute colour of the piece drops out and a
white rook and a black rook can train the same class. One calibration photo
therefore contributes 32 samples across 6 classes rather than 32 across 12,
and the model spends none of its capacity on a colour question the photometric
clustering in :mod:`boardeye.squares` already answers reliably.

Be clear about how far that goes: it does *not* mean one colour teaches the
other. Training on white pieces alone and testing on black scores about 50%
against a 17% chance baseline on the test fixtures — real transfer, but
partial, and concentrated in pawns. The accuracy the pipeline actually
achieves comes from calibration photos containing both colours, not from
generalisation across them.

**The model is small on purpose.** A calibration photo yields 32 samples. That
is far too few for a neural network and about right for a linear model over a
low-dimensional projection, which is why this is scikit-learn and not PyTorch.
The pipeline is scale -> PCA -> logistic regression: PCA because the raw
feature vector is much longer than the sample count, logistic regression
because it produces the per-class probabilities the editor uses to decide which
squares are worth your attention.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .squares import HEADROOM_RATIO
from .types import PIECE_TYPES

#: Every crop is resampled to this (width, height) before features are taken.
#: Taller than wide because piece crops carry headroom above the square.
CROP_SIZE = (32, 48)
#: Colour distance at which a pixel is considered fully "piece" rather than
#: background. Matches the tolerance used for occupancy so the two agree.
DEVIATION_SCALE = 96.0
#: How much of a crop's height is headroom over the square behind. Mirrors
#: :data:`boardeye.squares.HEADROOM_RATIO`, which governs how crops are cut.
HEADROOM_FRACTION = HEADROOM_RATIO / (1.0 + HEADROOM_RATIO)

DEFAULT_MODEL_DIR = Path.home() / ".boardeye"


class Recognizer(Protocol):
    """What the pipeline needs from a piece-type recogniser.

    Kept deliberately narrow so an entirely different backend could be dropped
    in later without touching detection, square analysis, or the editor.
    """

    def predict(self, crop: np.ndarray) -> dict[str, float]:
        """Map piece-type letters to probabilities for one square crop."""


def piece_map(crop: np.ndarray) -> np.ndarray:
    """Reduce a BGR crop to a map of where the piece is, ignoring its colour.

    A crop is not one flat background. It covers the piece's own square plus
    headroom above it, and that headroom belongs to the square behind — which
    on a chessboard is always the opposite colour. Measuring one background
    across the whole crop lands on a median between the two, and then *both*
    squares read as "not background" and the map lights up everywhere,
    drowning out the piece.

    So the two bands get their own reference, each taken from its own left and
    right edges where a centred piece does not reach.
    """
    resized = cv2.resize(crop, CROP_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    h, w = resized.shape[:2]
    band = max(1, w // 8)
    split = max(1, min(h - 1, int(round(h * HEADROOM_FRACTION))))

    deviation = np.empty((h, w), dtype=np.float32)
    for y0, y1 in ((0, split), (split, h)):
        region = resized[y0:y1]
        edges = np.concatenate(
            [region[:, :band].reshape(-1, 3), region[:, -band:].reshape(-1, 3)]
        )
        background = np.median(edges, axis=0)
        deviation[y0:y1] = np.linalg.norm(region - background[None, None, :], axis=2)

    # Rescale so the strongest piece pixels always land near 1.0. Without this
    # the map's overall brightness tracks how much the piece contrasts with its
    # square, which changes with lighting, focus and camera — and the model
    # then keys on contrast rather than shape. The floor stops a crop that is
    # nearly all background from having its noise amplified to full scale.
    strongest = float(np.percentile(deviation, 97))
    scale = max(strongest, DEVIATION_SCALE * 0.35)
    return np.clip(deviation / scale, 0.0, 1.0).astype(np.float32)


def _orientation_histogram(field: np.ndarray, cells=(4, 3), bins: int = 8) -> np.ndarray:
    """A small HOG-like descriptor over the piece map."""
    gx = cv2.Sobel(field, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(field, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    angle = np.mod(np.arctan2(gy, gx), np.pi)  # unsigned orientation

    h, w = field.shape
    rows, cols = cells
    out = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * h // rows, (r + 1) * h // rows
            x0, x1 = c * w // cols, (c + 1) * w // cols
            block_angle = angle[y0:y1, x0:x1].ravel()
            block_mag = magnitude[y0:y1, x0:x1].ravel()
            hist, _ = np.histogram(
                block_angle, bins=bins, range=(0.0, np.pi), weights=block_mag
            )
            norm = np.linalg.norm(hist)
            out.append(hist / norm if norm > 1e-6 else hist)
    return np.concatenate(out)


def extract_features(crop: np.ndarray) -> np.ndarray:
    """Feature vector for one square crop.

    Three complementary views: the coarse silhouette, its gradient structure,
    and the width-at-each-height profile. That last one carries most of the
    signal for chess pieces — a rook is a column of near-constant width, a
    bishop tapers to a point, a king is tall with a cross on top — and it is
    cheap and stable under the blur and noise of a phone photo.
    """
    field = piece_map(crop)

    silhouette = cv2.resize(field, (16, 24), interpolation=cv2.INTER_AREA).ravel()
    gradients = _orientation_histogram(field)
    width_profile = field.mean(axis=1)  # mass at each height
    height_profile = field.mean(axis=0)  # mass at each horizontal offset
    total_mass = np.array([field.mean(), field.max(), float((field > 0.5).mean())])

    return np.concatenate(
        [silhouette, gradients, width_profile, height_profile, total_mass]
    ).astype(np.float32)


def augment(crop: np.ndarray) -> list[np.ndarray]:
    """Plausible variations of one training crop.

    A single calibration photo gives 16 pawns but only 2 queens and 2 kings, so
    the rarest classes are the ones most likely to be misread — and in practice
    they are. Augmentation cannot invent genuine diversity, but small shifts
    and a mirror do reflect real variation: a piece is never centred on its
    square twice the same way, and a board photographed from the other side
    presents mirrored pieces.

    Border pixels are replicated rather than filled, because
    :func:`piece_map` reads the crop's border to estimate the background and a
    black fill would read as a piece.
    """
    h, w = crop.shape[:2]
    variants: list[np.ndarray] = []

    for source in (crop, cv2.flip(crop, 1)):
        variants.append(source)

        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            variants.append(
                cv2.warpAffine(source, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
            )

        for scale in (0.94, 1.06):
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
            variants.append(
                cv2.warpAffine(source, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
            )

        # Softened and washed-out copies. People calibrate once in good light
        # and then scan whenever a game ends — often handheld, often dimmer.
        # Without these the model learns the sharpness of the calibration photo
        # as if it were part of what a rook looks like, and a blurrier scan
        # reads pawns as rooks. Measured: this is the difference between ~70%
        # and ~95% when calibration and scan conditions disagree.
        for sigma in (0.7, 1.4):
            variants.append(cv2.GaussianBlur(source, (0, 0), sigma))
        faded = (source.astype(np.float32) - 128.0) * 0.75 + 128.0
        variants.append(np.clip(faded, 0, 255).astype(np.uint8))

    return variants


@dataclass
class TrainingStats:
    """What the model was trained on, so the CLI can report it honestly."""

    samples: int
    per_class: dict[str, int]
    cross_val_accuracy: float | None = None

    def summary(self) -> str:
        counts = ", ".join(f"{k}:{v}" for k, v in sorted(self.per_class.items()))
        line = f"{self.samples} samples ({counts})"
        if self.cross_val_accuracy is not None:
            line += f" — held-out accuracy {self.cross_val_accuracy:.0%}"
        return line


class PieceClassifier:
    """A piece-type model plus the crops it was trained on.

    The training crops are kept alongside the fitted model so the dataset
    survives changes to the feature extractor: adding a feature means
    retraining from stored images rather than asking you to re-photograph
    anything.
    """

    def __init__(self, model_dir: Path | str | None = None):
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.pipeline: Pipeline | None = None
        self.labels: list[str] = []
        self._crops: list[np.ndarray] = []
        self._targets: list[str] = []
        self.stats: TrainingStats | None = None

    # ---------------------------------------------------------------- data

    @property
    def dataset_path(self) -> Path:
        return self.model_dir / "dataset.npz"

    @property
    def model_path(self) -> Path:
        return self.model_dir / "model.pkl"

    @property
    def is_trained(self) -> bool:
        return self.pipeline is not None

    @property
    def sample_count(self) -> int:
        return len(self._targets)

    def add_samples(self, crops: list[np.ndarray], labels: list[str]) -> None:
        """Append labelled crops. ``labels`` are piece-type letters (``pnbrqk``)."""
        if len(crops) != len(labels):
            raise ValueError("crops and labels must be the same length")
        for crop, label in zip(crops, labels):
            label = label.lower()
            if label not in PIECE_TYPES:
                raise ValueError(f"not a piece type: {label!r}")
            # Store the normalised crop, small enough that thousands cost little.
            self._crops.append(cv2.resize(crop, CROP_SIZE, interpolation=cv2.INTER_AREA))
            self._targets.append(label)

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for label in self._targets:
            counts[label] = counts.get(label, 0) + 1
        return counts

    # --------------------------------------------------------------- train

    def train(self) -> TrainingStats:
        """Fit the model on everything collected so far."""
        counts = self.class_counts()
        if len(counts) < 2:
            raise ValueError(
                "need at least two different piece types to train; "
                f"have {sorted(counts) or 'nothing'}"
            )

        train_crops: list[np.ndarray] = []
        train_targets: list[str] = []
        for crop, label in zip(self._crops, self._targets):
            for variant in augment(crop):
                train_crops.append(variant)
                train_targets.append(label)

        features = np.stack([extract_features(crop) for crop in train_crops])
        targets = np.array(train_targets)

        n_samples = len(targets)
        # PCA cannot keep more components than it has samples or raw features.
        components = int(min(24, n_samples - 1, features.shape[1]))
        self.pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=components, random_state=0)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        # A calibration photo has 16 pawns and 2 kings; without
                        # rebalancing the model would learn to answer "pawn".
                        class_weight="balanced",
                        C=1.0,
                        random_state=0,
                    ),
                ),
            ]
        )
        self.pipeline.fit(features, targets)
        self.labels = list(self.pipeline.named_steps["model"].classes_)

        # Cross-validate on the originals only. Scoring augmented copies would
        # put near-duplicates of a training crop into the test fold and report
        # an accuracy that is mostly memorisation.
        originals = np.stack([extract_features(crop) for crop in self._crops])
        self.stats = TrainingStats(
            samples=len(self._targets),
            per_class=counts,
            cross_val_accuracy=self._cross_val(originals, np.array(self._targets)),
        )
        return self.stats

    @staticmethod
    def _cross_val(features: np.ndarray, targets: np.ndarray) -> float | None:
        """Held-out accuracy, when there is enough data for it to mean anything.

        Returned as a number for the CLI to print, with the important caveat
        that it measures performance on crops from the *same* photographs it
        trained on. It says the model learned something; it does not predict
        accuracy on a new photo taken in different light.
        """
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        counts = {label: int((targets == label).sum()) for label in set(targets)}
        if min(counts.values()) < 2 or len(counts) < 2:
            return None
        folds = min(3, min(counts.values()))
        try:
            pipeline = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "pca",
                        PCA(
                            n_components=int(
                                min(16, len(targets) - len(targets) // folds - 1)
                            ),
                            random_state=0,
                        ),
                    ),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000, class_weight="balanced", random_state=0
                        ),
                    ),
                ]
            )
            scores = cross_val_score(
                pipeline,
                features,
                targets,
                cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=0),
            )
            return float(scores.mean())
        except ValueError:
            return None

    # ------------------------------------------------------------- predict

    def predict(self, crop: np.ndarray) -> dict[str, float]:
        """Probabilities over piece types for one crop, highest first."""
        if self.pipeline is None:
            raise RuntimeError("classifier is not trained yet — run 'boardeye calibrate'")
        features = extract_features(crop).reshape(1, -1)
        probabilities = self.pipeline.predict_proba(features)[0]
        ranked = sorted(
            zip(self.labels, probabilities), key=lambda item: item[1], reverse=True
        )
        return {label: float(p) for label, p in ranked}

    # ---------------------------------------------------------- persistence

    def save(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if self._crops:
            np.savez_compressed(
                self.dataset_path,
                crops=np.stack(self._crops),
                targets=np.array(self._targets),
            )
        if self.pipeline is not None:
            with open(self.model_path, "wb") as handle:
                pickle.dump(
                    {"pipeline": self.pipeline, "labels": self.labels, "stats": self.stats},
                    handle,
                )

    def load(self) -> bool:
        """Load dataset and model if present. Returns True if a model was loaded."""
        if self.dataset_path.exists():
            data = np.load(self.dataset_path, allow_pickle=False)
            self._crops = list(data["crops"])
            self._targets = [str(t) for t in data["targets"]]
        if self.model_path.exists():
            with open(self.model_path, "rb") as handle:
                blob = pickle.load(handle)
            self.pipeline = blob["pipeline"]
            self.labels = blob["labels"]
            self.stats = blob.get("stats")
            return True
        return False


def load_classifier(model_dir: Path | str | None = None) -> PieceClassifier:
    """Load the on-disk classifier, trained or not."""
    classifier = PieceClassifier(model_dir)
    classifier.load()
    return classifier
