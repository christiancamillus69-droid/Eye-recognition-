"""Point a webcam at the board and let it grab its own frame.

Handheld or tripod, a live feed gives you many attempts per second where a
still photo gives you one. The scanner exploits that by refusing to commit
until the board has been detected in the *same place* for several consecutive
frames — which filters out motion blur, a hand crossing the view, and the
half-second while autofocus settles, all of which produce a detection that
jumps around before it settles.

**Not verified.** This container has no camera, so unlike the rest of the
package this module has never been run against real hardware. The geometry it
depends on is tested; the capture loop around it is not.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .detect import BoardNotFound, Detection, detect_board

#: Corners must stay within this many pixels across frames to count as steady.
STABILITY_TOLERANCE = 12.0
#: Consecutive steady frames required before the frame is accepted.
STABLE_FRAMES = 8


class CameraUnavailable(Exception):
    """The requested camera could not be opened."""


def preview_supported() -> bool:
    """Whether this OpenCV build can open a window.

    ``opencv-python-headless`` — the default dependency, because it installs
    everywhere without system GUI libraries — has no ``imshow``. Detecting that
    here lets the scanner explain the situation instead of crashing.
    """
    return hasattr(cv2, "imshow") and callable(getattr(cv2, "imshow", None))


def _corner_shift(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Largest distance any corner moved between two detections."""
    return max(float(np.hypot(p[0] - q[0], p[1] - q[1])) for p, q in zip(a, b))


@dataclass
class Capture:
    """A frame the scanner decided to keep."""

    frame: np.ndarray
    detection: Detection
    steady_frames: int


def _annotate(frame: np.ndarray, detection: Detection | None, steady: int) -> np.ndarray:
    """Draw the detected board and a lock-in progress bar onto a preview frame."""
    canvas = frame.copy()
    if detection is not None:
        colour = (80, 220, 120) if detection.confident else (60, 160, 240)
        points = np.int32(detection.corners).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], True, colour, 3)
        cv2.putText(
            canvas,
            f"score {detection.score:.2f}  steady {steady}/{STABLE_FRAMES}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colour,
            2,
        )
    else:
        cv2.putText(
            canvas,
            "no board found - fill more of the frame",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (60, 60, 230),
            2,
        )
    cv2.putText(
        canvas,
        "SPACE capture now   Q quit",
        (12, canvas.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
    )
    return canvas


def scan_live(
    camera_index: int = 0,
    *,
    stable_frames: int = STABLE_FRAMES,
    tolerance: float = STABILITY_TOLERANCE,
    preview: bool = True,
    timeout: float | None = 120.0,
) -> Capture:
    """Watch a camera until the board holds still, then return that frame.

    Set ``preview=False`` for a build of OpenCV without GUI support; the
    scanner then reports progress on the terminal and captures automatically.
    """
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise CameraUnavailable(
            f"could not open camera {camera_index}. "
            "Check it is connected and not in use by another application."
        )

    show = preview and preview_supported()
    if preview and not show:
        print(
            "This OpenCV build has no display support, so there is no live preview.\n"
            "Capturing automatically once the board holds still. For a preview, "
            "swap the headless package:\n"
            "    pip uninstall -y opencv-python-headless && pip install opencv-python"
        )

    previous: Detection | None = None
    steady = 0
    started = time.monotonic()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise CameraUnavailable("camera stopped returning frames")

            try:
                detection = detect_board(frame)
            except BoardNotFound:
                detection = None

            if detection is not None and detection.confident:
                if previous is not None and _corner_shift(
                    detection.corners, previous.corners
                ) <= tolerance:
                    steady += 1
                else:
                    steady = 1
                previous = detection
            else:
                steady = 0
                previous = None

            if show:
                cv2.imshow("BoardEye — live scan", _annotate(frame, detection, steady))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    raise KeyboardInterrupt("scan cancelled")
                if key == ord(" ") and detection is not None:
                    return Capture(frame.copy(), detection, steady)
            elif steady and steady % 4 == 0:
                print(f"  board steady for {steady}/{stable_frames} frames…")

            if previous is not None and steady >= stable_frames:
                return Capture(frame.copy(), previous, steady)

            if timeout is not None and time.monotonic() - started > timeout:
                raise TimeoutError(
                    f"no steady board within {timeout:.0f}s. Try more even lighting, "
                    "or fill more of the frame with the board."
                )
    finally:
        capture.release()
        if show:
            cv2.destroyAllWindows()
