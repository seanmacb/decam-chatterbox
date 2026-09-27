"""Command-line interface.

Subcommands
-----------
``serve``
    Consume the configured source (SCIMMA by default) and post about each
    notice.
``replay``
    Handle one or more saved notice files. ``--dry-run`` renders everything
    and prints the message without posting; this is the main development
    loop.
``test-post``
    Send a short message to confirm Slack credentials and channel access.
``doctor``
    Report which capabilities are available and how to fix the missing ones.
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import Config, load_config

__all__ = ["main", "build_parser"]

logger = logging.getLogger("decam_chatterbox")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="decam-chatterbox",
        description="Low-latency Slack reporting of gravitational-wave alerts for DECam.",
    )
    parser.add_argument("-c", "--config", help="Path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log at DEBUG level")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("serve", help="Consume the configured alert source (SCIMMA by default)")

    p_replay = sub.add_parser("replay", help="Handle one or more saved notice files")
    p_replay.add_argument("paths", nargs="+", help="Record files (.json or .avro)")
    p_replay.add_argument(
        "--dry-run",
        action="store_true",
        help="Render and print the message without posting to Slack",
    )
    p_replay.add_argument("--out-dir", help="Where to write the dark-hours plot")

    sub.add_parser("test-post", help="Post a test message to confirm Slack access")
    sub.add_parser("doctor", help="Report what works, what does not, and how to fix it")

    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # These are noisy at INFO and say nothing useful about our own work.
    #
    # Deliberately excludes "astropy": astropy installs its own Logger
    # subclass at import time and fails if a plain logger of that name
    # already exists, so calling getLogger("astropy") here would break
    # importing astropy.
    for noisy in ("matplotlib", "urllib3", "numexpr", "PIL", "fsspec", "healpy", "hop"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _cmd_serve(args: argparse.Namespace, config: Config) -> int:
    from .app import run_service

    try:
        handled = run_service(config)
    except Exception as exc:
        # run_service has already posted this to the channel; the operator
        # gets one line and a non-zero exit, not a traceback, since the
        # traceback is in the log with the rest of the context.
        print(f"Monitoring stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Handled {handled} record(s).")
    return 0


def _cmd_replay(args: argparse.Namespace, config: Config) -> int:
    from .app import process_notice
    from .ingest.decode import load_record_file
    from .slackbot.client import SlackPoster, render_blocks_as_text

    poster = SlackPoster(config, dry_run=args.dry_run)
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else None

    failures = 0
    for path in args.paths:
        print(f"\n{'=' * 72}\n{path}\n{'=' * 72}")
        try:
            record = load_record_file(path)
            report = process_notice(
                record,
                config,
                poster=poster,
                post=not args.dry_run,
                out_dir=out_dir,
            )
        except Exception as exc:
            logger.error("Failed on %s: %s", path, exc, exc_info=args.verbose)
            failures += 1
            continue

        if args.dry_run:
            print(render_blocks_as_text(report.blocks))

        print(f"Handled in {report.elapsed_s:.2f} s")
        for plot in report.plots:
            print(f"  plot: {plot}")
        for warning in report.warnings:
            print(f"  warning: {warning}")
        if report.posted is not None and report.posted.offline:
            print("  (offline: payload written instead of posted)")

    return 1 if failures else 0


def _cmd_test_post(args: argparse.Namespace, config: Config) -> int:
    from .slackbot.client import SlackPoster

    poster = SlackPoster(config)
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": ":wave: *decam-chatterbox test post* -- if you can read this, credentials and "
                "channel access are working.",
            },
        }
    ]
    posted = poster.post(blocks, "decam-chatterbox test post", label="test_post")
    if posted.offline:
        print(
            f"Offline: {config.slack.bot_token_env} is not set, so nothing was sent. "
            f"Payload written under {poster.output_dir}."
        )
        return 1
    print(f"Posted to {posted.channel} (ts={posted.ts}).")
    return 0


def _cmd_doctor(args: argparse.Namespace, config: Config) -> int:
    from .doctor import diagnose, format_report

    checks = diagnose(config)
    print(format_report(checks))
    return 1 if any(not c.ok and c.fatal for c in checks) else 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``decam-chatterbox`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"Could not load configuration: {exc}", file=sys.stderr)
        return 2

    handlers = {
        "serve": _cmd_serve,
        "replay": _cmd_replay,
        "test-post": _cmd_test_post,
        "doctor": _cmd_doctor,
    }
    return handlers[args.command](args, config)


if __name__ == "__main__":
    sys.exit(main())
