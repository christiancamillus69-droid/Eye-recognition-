"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import cv2
import numpy as np

from . import export
from .calibrate import CalibrationError, calibrate_from_image
from .classifier import PieceClassifier, load_classifier
from .detect import BoardNotFound
from .fen import material_warnings, validate
from .pipeline import read_position
from .squares import piece_crop
from .types import BoardReading, square_name


def _load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise SystemExit(f"could not read an image from {path!r}")
    return image


def _dump_debug(directory: Path, image: np.ndarray, reading: BoardReading) -> None:
    """Write the intermediate images, for working out why a read went wrong."""
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / "01-source.png"), image)
    if reading.warped is not None:
        cv2.imwrite(str(directory / "02-rectified.png"), reading.warped)
        crops = directory / "squares"
        crops.mkdir(exist_ok=True)
        for r in range(8):
            for f in range(8):
                square = reading.squares[r][f]
                if not square.occupied:
                    continue
                label = f"{square_name(r, f)}-{square.piece or 'unknown'}"
                cv2.imwrite(str(crops / f"{label}.png"), piece_crop(reading.warped, r, f))
    print(f"debug images written to {directory}/")


def _report_reading(reading: BoardReading, fen: str) -> None:
    flagged = [
        (r, f)
        for r, f in reading.review_order()
        if reading.squares[r][f].occupied and reading.squares[r][f].review_priority < 0.6
    ]
    print(f"FEN      {fen}")
    print(f"Lichess  {export.lichess_url(fen)}")
    print(f"Chess.com {export.chesscom_url(fen)}")
    problems = validate(fen) + material_warnings(reading.grid())
    if problems:
        print("\nCheck this position:")
        for problem in problems:
            print(f"  - {problem}")
    if flagged:
        names = ", ".join(square_name(r, f) for r, f in flagged[:8])
        print(f"\n{len(flagged)} square(s) read with low confidence: {names}")
        print("Run without --fen-only to review them before trusting an engine.")


# --------------------------------------------------------------- subcommands


def cmd_calibrate(args: argparse.Namespace) -> int:
    classifier = load_classifier(args.model_dir)
    total = 0
    for path in args.photos:
        image = _load_image(path)
        try:
            result = calibrate_from_image(image, classifier)
        except (CalibrationError, BoardNotFound) as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            return 1
        total += result.samples_added
        turn_note = (
            f", rotated {result.rotation_applied * 90}° to put White at the bottom"
            if result.rotation_applied
            else ""
        )
        print(f"{path}: learned {result.samples_added} pieces{turn_note}")
        for warning in result.warnings:
            print(f"  note: {warning}")

    print(f"\nModel now holds {classifier.sample_count} samples.")
    if classifier.stats:
        print(f"  {classifier.stats.summary()}")
    if len(args.photos) == 1:
        print(
            "\nOne photo is enough if it is a good one. What matters most is that the\n"
            "calibration photo is sharp, evenly lit and fills the frame — on the test\n"
            "fixtures a good single photo reached 85% piece accuracy while a poor one\n"
            "managed 68%.\n"
            "Adding a second photo is cheap insurance: it barely improves on a good\n"
            "first photo, but it pulls a bad one back up to about 84%."
        )
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    classifier = load_classifier(args.model_dir)
    if not classifier.is_trained:
        print(
            "No trained model found, so piece types cannot be read yet — every piece\n"
            "will come through as a pawn for you to correct.\n"
            "Run 'boardeye calibrate <photo of the starting position>' first.\n",
            file=sys.stderr,
        )

    if args.camera is not None:
        from .camera import CameraUnavailable, scan_live

        try:
            capture = scan_live(args.camera, preview=not args.no_preview)
        except (CameraUnavailable, TimeoutError) as exc:
            print(f"live scan failed: {exc}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("cancelled")
            return 130
        image = capture.frame
        print(f"captured a frame (detection score {capture.detection.score:.2f})")
    else:
        image = _load_image(args.photo)

    try:
        reading = read_position(image, classifier)
    except BoardNotFound as exc:
        print(
            f"could not find a board in that image: {exc}\n"
            "If the board is there, open the editor and click its four corners.",
            file=sys.stderr,
        )
        return 1

    from .fen import Position

    fen = Position(grid=reading.grid()).fen()

    if args.debug:
        _dump_debug(Path(args.debug), image, reading)

    if args.fen_only:
        _report_reading(reading, fen)
        return 0

    if args.open:
        url = export.lichess_url(fen) if args.open == "lichess" else export.chesscom_url(fen)
        print(f"opening {url}")
        webbrowser.open(url)
        return 0

    from .editor.app import EditorSession, serve

    session = EditorSession(
        image=image,
        reading=reading,
        classifier=classifier,
        learn_from_corrections=not args.no_learn,
    )
    try:
        serve(session, host=args.host, port=args.port, open_browser=not args.no_browser)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Grade photos before the user commits to them."""
    from .quality import holdout_accuracy, inspect

    images = []
    reports = []
    for path in args.photos:
        image = _load_image(path)
        images.append(image)
        reports.append(inspect(image, label=path, expect_start=not args.position))

    for report in reports:
        status = "good" if report.usable and report.score >= 0.75 else (
            "usable" if report.usable else "PROBLEM"
        )
        print(f"\n{report.label}  [{status}]")
        if report.found_board:
            print(
                f"  detection {report.score:.2f} | sharpness {report.sharpness:.0f} | "
                f"board fills {report.fill:.0%} of frame | clipped {report.clipping:.1%}"
            )
        for problem in report.problems:
            print(f"  ! {problem}")
        for tip in report.advice:
            print(f"  - {tip}")
        if not report.problems and not report.advice:
            print("  nothing to improve")

    usable = [image for image, report in zip(images, reports) if report.usable]
    if not args.position and len(usable) >= 2:
        print("\nMeasuring how well this will actually work on your board…")
        print("(training on all but one photo, then reading the one left out)")
        result = holdout_accuracy(usable)
        if result is None:
            print("  could not complete the held-out check")
        else:
            print(f"\n  {result.summary()}")
            weak = [
                f"{piece}: {hit}/{n}"
                for piece, (hit, n) in result.per_type.items()
                if n and hit / n < 0.75
            ]
            if weak:
                print(f"  weakest piece types — {', '.join(weak)}")
                print(
                    "  Those are the ones to watch in the editor; correcting them "
                    "there feeds straight back into the model."
                )
            if result.accuracy < 0.7:
                print(
                    "\n  That is low. Usually it means the calibration photos are "
                    "soft, unevenly lit, or shot from too low an angle."
                )
    elif not args.position and len(usable) == 1:
        print(
            "\nGive two or more starting-position photos and this will measure the "
            "accuracy you can expect on your own board."
        )
    return 0 if all(r.usable for r in reports) else 1


def cmd_retrain(args: argparse.Namespace) -> int:
    classifier = load_classifier(args.model_dir)
    if not classifier.sample_count:
        print(
            "Nothing to retrain from. Run 'boardeye calibrate' first.", file=sys.stderr
        )
        return 1
    stats = classifier.train()
    classifier.save()
    print(f"retrained: {stats.summary()}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    classifier = PieceClassifier(args.model_dir)
    loaded = classifier.load()
    print(f"model directory  {classifier.model_dir}")
    print(f"trained model    {'yes' if loaded else 'no'}")
    print(f"stored samples   {classifier.sample_count}")
    if classifier.sample_count:
        counts = ", ".join(f"{k}:{v}" for k, v in sorted(classifier.class_counts().items()))
        print(f"per piece type   {counts}")
    if classifier.stats and classifier.stats.cross_val_accuracy is not None:
        print(
            f"held-out score   {classifier.stats.cross_val_accuracy:.0%} "
            "(on calibration photos, not a prediction for new ones)"
        )
    if not loaded:
        print("\nRun: boardeye calibrate <photo of the starting position>")
    return 0


# -------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="boardeye",
        description="Photograph an over-the-board position and get an engine-ready FEN.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="where the trained model and its crops live (default: ~/.boardeye)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="learn your pieces from photos of the starting position",
        description=(
            "Photograph your board set up for a new game. Because the layout is "
            "known, every piece labels itself and no annotation is needed."
        ),
    )
    calibrate.add_argument("photos", nargs="+", help="photo(s) of the starting position")
    calibrate.set_defaults(func=cmd_calibrate)

    scan = subparsers.add_parser(
        "scan", help="read a position from a photo or a live camera"
    )
    source = scan.add_mutually_exclusive_group(required=True)
    source.add_argument("photo", nargs="?", help="photograph of the position")
    source.add_argument(
        "--camera", type=int, metavar="INDEX", help="capture from this camera instead"
    )
    scan.add_argument(
        "--fen-only", action="store_true", help="print the FEN and links, skip the editor"
    )
    scan.add_argument(
        "--open",
        choices=("lichess", "chesscom"),
        help="open the position in that site immediately, skipping review",
    )
    scan.add_argument("--debug", metavar="DIR", help="write intermediate images here")
    scan.add_argument(
        "--no-learn", action="store_true", help="do not feed corrections back into the model"
    )
    scan.add_argument("--no-browser", action="store_true", help="do not open a browser")
    scan.add_argument("--no-preview", action="store_true", help="live scan without a window")
    scan.add_argument("--host", default="127.0.0.1")
    scan.add_argument("--port", type=int, default=5000)
    scan.set_defaults(func=cmd_scan)

    check = subparsers.add_parser(
        "check",
        help="grade photos before relying on them",
        description=(
            "Measures each photo and says what to change. Given two or more "
            "photos of the starting position, it also trains on all but one and "
            "reads the one left out, which measures the accuracy you can expect "
            "on your own board rather than on test fixtures."
        ),
    )
    check.add_argument("photos", nargs="+", help="photo(s) to grade")
    check.add_argument(
        "--position",
        action="store_true",
        help="these are ordinary position photos, not the starting position",
    )
    check.set_defaults(func=cmd_check)

    retrain = subparsers.add_parser(
        "retrain", help="refit the model from stored crops without new photos"
    )
    retrain.set_defaults(func=cmd_retrain)

    status = subparsers.add_parser("status", help="show what the model has learned")
    status.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
