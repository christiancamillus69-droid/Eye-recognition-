"""Find the board in a photograph and flatten it to a top-down square.

Two independent strategies run and compete, because neither is reliable alone:
a contour search (good when the board contrasts with the table) and a Hough
line search (good when the board fills the frame or blends into its
surroundings, since it keys on the internal grid rather than the outer edge).

Both are scored by the same test — warp the candidate, then measure how much
the result actually looks like a checkerboard — and the winner is returned.
That scoring step matters more than either detector: it is what lets the code
*know* it found a board rather than merely assert it, and it is what tells the
CLI when to fall back to asking you to click the corners.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

BOARD_PX = 640

#: Below this checkerboard score, a detection is not trustworthy enough to use
#: without the user confirming the corners. Calibrated against the synthetic
#: fixtures plus a margin; see tests/test_detect.py.
MIN_ACCEPTABLE_SCORE = 0.35


class BoardNotFound(Exception):
    """Raised when no candidate looked enough like a chessboard."""


@dataclass
class Detection:
    """A located board, ready to be cut into squares."""

    corners: list[tuple[float, float]]
    warped: np.ndarray
    method: str
    score: float

    @property
    def confident(self) -> bool:
        return self.score >= MIN_ACCEPTABLE_SCORE


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left.

    Uses coordinate sums and differences, which is orientation-stable for any
    convex quadrilateral that is not rotated more than ~45 degrees — beyond
    that "top-left" stops being meaningful anyway, and the editor's rotate
    button is the right fix.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  # top-left  has the smallest x+y
    ordered[2] = pts[np.argmax(s)]  # bottom-right the largest
    d = np.diff(pts, axis=1).ravel()  # y - x
    ordered[1] = pts[np.argmin(d)]  # top-right
    ordered[3] = pts[np.argmax(d)]  # bottom-left
    return ordered


def warp_from_corners(
    image: np.ndarray, corners, board_px: int = BOARD_PX
) -> np.ndarray:
    """Rectify the quadrilateral ``corners`` into a square top-down board."""
    src = order_corners(np.asarray(corners, dtype=np.float32))
    dst = np.float32(
        [[0, 0], [board_px, 0], [board_px, board_px], [0, board_px]]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (board_px, board_px))


def checkerboard_score(warped: np.ndarray) -> float:
    """How strongly a rectified image alternates like a chessboard, in [0, 1].

    Correlates each square's median brightness against the ideal +1/-1 checker
    pattern. Pieces sitting on squares weaken but do not destroy the signal:
    roughly half the squares are empty in any real position and they carry it.

    The median (not the mean) is deliberate — a piece occupying part of a
    square shifts a mean substantially but often leaves the median on the
    square's own colour.
    """
    grey = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
    size = grey.shape[0] // 8
    if size < 4:
        return 0.0

    medians = np.empty(64, dtype=np.float64)
    pattern = np.empty(64, dtype=np.float64)
    for r in range(8):
        for f in range(8):
            cell = grey[r * size : (r + 1) * size, f * size : (f + 1) * size]
            medians[r * 8 + f] = float(np.median(cell))
            # a1 is dark and sits at row 7 file 0, giving parity (r + f) % 2.
            pattern[r * 8 + f] = -1.0 if (r + f) % 2 else 1.0

    if np.std(medians) < 1e-6:
        return 0.0
    correlation = np.corrcoef(medians, pattern)[0, 1]
    return float(abs(correlation)) if np.isfinite(correlation) else 0.0


def _quad_from_contours(image: np.ndarray) -> list[np.ndarray]:
    """Candidate quads from the largest four-sided contours in the image."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (5, 5), 0)
    edges = cv2.Canny(grey, 40, 140)
    # Close gaps where a piece overlaps the board's outer edge, so the outline
    # survives as one contour instead of breaking into arcs.
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = image.shape[0] * image.shape[1]
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.05:
            continue
        perimeter = cv2.arcLength(contour, True)
        for epsilon in (0.02, 0.04, 0.08):
            approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                candidates.append((area, approx.reshape(4, 2).astype(np.float32)))
                break

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [quad for _, quad in candidates[:5]]


def _cluster_by_angle(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split Hough lines into the board's two families of parallel lines."""
    thetas = lines[:, 1]
    # Angles live on a circle of period pi, so cluster on the doubled angle to
    # keep lines near 0 and near pi together instead of at opposite extremes.
    doubled = np.stack([np.cos(2 * thetas), np.sin(2 * thetas)], axis=1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(doubled, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    return lines[labels == 0], lines[labels == 1]


def _extreme_lines(family: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two outermost lines of a parallel family, by signed offset."""
    # Normalise so that all lines in the family point the same way before
    # comparing rho, otherwise a line recorded with negative rho and flipped
    # theta looks like an outlier when it is actually in the middle.
    normalised = []
    reference = family[0, 1]
    for rho, theta in family:
        if abs(theta - reference) > np.pi / 2:
            rho, theta = -rho, theta - np.pi
        normalised.append((rho, theta))
    arr = np.array(normalised)
    return arr[np.argmin(arr[:, 0])], arr[np.argmax(arr[:, 0])]


def _intersect(line_a, line_b) -> tuple[float, float] | None:
    """Intersection of two lines given in Hough (rho, theta) form."""
    rho1, theta1 = line_a
    rho2, theta2 = line_b
    a = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([rho1, rho2])
    if abs(np.linalg.det(a)) < 1e-6:
        return None
    x, y = np.linalg.solve(a, b)
    return float(x), float(y)


def _quad_from_hough(image: np.ndarray) -> list[np.ndarray]:
    """Candidate quad from the outermost lines of the board's grid."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (5, 5), 0)
    edges = cv2.Canny(grey, 40, 140)

    smaller = min(image.shape[:2])
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=max(60, smaller // 8))
    if lines is None or len(lines) < 8:
        return []
    lines = lines.reshape(-1, 2)

    try:
        family_a, family_b = _cluster_by_angle(lines)
    except cv2.error:
        return []
    if len(family_a) < 2 or len(family_b) < 2:
        return []

    a_low, a_high = _extreme_lines(family_a)
    b_low, b_high = _extreme_lines(family_b)

    corners = []
    for line_a in (a_low, a_high):
        for line_b in (b_low, b_high):
            point = _intersect(line_a, line_b)
            if point is None:
                return []
            corners.append(point)

    quad = np.array(corners, dtype=np.float32)
    h, w = image.shape[:2]
    # Reject intersections that fall far outside the frame; those come from
    # nearly-parallel lines and produce a wildly wrong warp.
    if np.any(quad[:, 0] < -w) or np.any(quad[:, 0] > 2 * w):
        return []
    if np.any(quad[:, 1] < -h) or np.any(quad[:, 1] > 2 * h):
        return []
    return [quad]


def _refine_corners(
    image: np.ndarray, corners: np.ndarray, board_px: int, radius: int = 6
) -> tuple[np.ndarray, float]:
    """Nudge each corner a few pixels to maximise the checkerboard score.

    Detectors typically land within a handful of pixels of the true corner,
    which is enough to shift every square boundary and smear piece crops. A
    short coordinate-descent pass recovers that alignment cheaply.
    """
    best = order_corners(corners).copy()
    best_score = checkerboard_score(warp_from_corners(image, best, board_px))

    for _ in range(2):  # two sweeps is enough to converge in practice
        improved = False
        for index in range(4):
            for dx, dy in ((-radius, 0), (radius, 0), (0, -radius), (0, radius)):
                trial = best.copy()
                trial[index] += (dx, dy)
                score = checkerboard_score(warp_from_corners(image, trial, board_px))
                if score > best_score + 1e-4:
                    best, best_score = trial, score
                    improved = True
        if not improved:
            break
    return best, best_score


def detect_board(
    image: np.ndarray, board_px: int = BOARD_PX, refine: bool = True
) -> Detection:
    """Locate the board and return it rectified.

    Raises :class:`BoardNotFound` when nothing scored above zero. A returned
    detection may still be poor — check :attr:`Detection.confident` before
    trusting it, and fall back to manual corners when it is False.
    """
    if image is None or image.size == 0:
        raise BoardNotFound("empty image")

    candidates: list[tuple[np.ndarray, str]] = []
    candidates += [(quad, "contour") for quad in _quad_from_contours(image)]
    candidates += [(quad, "hough") for quad in _quad_from_hough(image)]
    # The whole frame, for photos cropped tight to the board already.
    h, w = image.shape[:2]
    candidates.append(
        (np.float32([[0, 0], [w, 0], [w, h], [0, h]]), "full-frame")
    )

    best: Detection | None = None
    for quad, method in candidates:
        try:
            ordered = order_corners(quad)
            warped = warp_from_corners(image, ordered, board_px)
        except cv2.error:
            continue
        score = checkerboard_score(warped)
        if best is None or score > best.score:
            best = Detection(
                corners=[(float(x), float(y)) for x, y in ordered],
                warped=warped,
                method=method,
                score=score,
            )

    if best is None or best.score <= 0.0:
        raise BoardNotFound("no candidate region looked like a chessboard")

    if refine:
        refined, refined_score = _refine_corners(
            image, np.array(best.corners, dtype=np.float32), board_px
        )
        if refined_score > best.score:
            best = Detection(
                corners=[(float(x), float(y)) for x, y in refined],
                warped=warp_from_corners(image, refined, board_px),
                method=f"{best.method}+refined",
                score=refined_score,
            )

    return best


def detect_from_corners(
    image: np.ndarray, corners, board_px: int = BOARD_PX
) -> Detection:
    """Build a Detection from corners the user clicked in the editor."""
    ordered = order_corners(np.asarray(corners, dtype=np.float32))
    warped = warp_from_corners(image, ordered, board_px)
    return Detection(
        corners=[(float(x), float(y)) for x, y in ordered],
        warped=warped,
        method="manual",
        score=checkerboard_score(warped),
    )
