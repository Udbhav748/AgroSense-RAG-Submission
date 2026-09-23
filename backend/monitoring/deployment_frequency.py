"""Deployment Frequency: number of successful production releases in a
given period.

"Release" = an annotated git tag, matching the tagging convention
docs/OPERATIONS.md's Rollback plan already establishes (`git tag -a
v0.1.0 -m "known-good: ..."`) — the only release-marking mechanism this
project actually has (no deploy platform webhook, no release log). Reads
tags via git itself rather than a JSON log file the way the other
monitoring/*.py scripts do, since a release isn't a runtime event this
app's own logging could ever observe.

Usage (from anywhere inside the repo):
    python monitoring/deployment_frequency.py                    # all tags, all time
    python monitoring/deployment_frequency.py --days 30           # trailing 30 days
    python monitoring/deployment_frequency.py --since 2026-01-01 --until 2026-06-30
    python monitoring/deployment_frequency.py --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime, timedelta


def _run_git(args: list[str]) -> str:
    """Run a git command and return its stdout. git itself resolves the
    repo root regardless of the current working directory within it, so
    this works whether invoked from backend/ or anywhere else inside the
    tree. Raises SystemExit with a clear message (not a raw traceback) if
    git is missing or the command fails."""
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise SystemExit("git is not installed or not on PATH.") from None
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"git {' '.join(args)} failed: {exc.stderr.strip()}") from None
    return result.stdout


def list_releases() -> list[dict]:
    """Every tag, with its creation date and subject line (empty for a
    lightweight tag with no message). %(creatordate) is the tag object's
    own date for an annotated tag, or the pointed-at commit's date for a
    lightweight one — either way, "when this release was marked," not
    necessarily when the code was written."""
    output = _run_git(
        ["for-each-ref", "refs/tags", "--format=%(refname:short)|%(creatordate:iso-strict)|%(subject)"]
    )
    releases = []
    for line in output.splitlines():
        if not line.strip():
            continue
        name, date_str, subject = line.split("|", 2)
        try:
            date = datetime.fromisoformat(date_str)
        except ValueError:
            continue
        releases.append({"tag": name, "date": date, "subject": subject})
    return releases


def filter_window(releases: list[dict], since: datetime | None, until: datetime | None) -> list[dict]:
    return [
        release
        for release in releases
        if (since is None or release["date"] >= since) and (until is None or release["date"] <= until)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=None, help="Only count releases in the trailing N days.")
    parser.add_argument("--since", default=None, help="Only count releases on/after this date (YYYY-MM-DD).")
    parser.add_argument("--until", default=None, help="Only count releases on/before this date (YYYY-MM-DD).")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human text.")
    args = parser.parse_args()

    if args.days is not None and (args.since or args.until):
        raise SystemExit("Use --days, or --since/--until, not both.")

    since: datetime | None = None
    until: datetime | None = None
    if args.days is not None:
        since = datetime.now(UTC) - timedelta(days=args.days)
    if args.since:
        since = datetime.fromisoformat(args.since).replace(tzinfo=UTC)
    if args.until:
        until = datetime.fromisoformat(args.until).replace(tzinfo=UTC)

    all_releases = list_releases()
    windowed = filter_window(all_releases, since, until)
    windowed.sort(key=lambda release: release["date"], reverse=True)

    period_days = None
    if since is not None:
        period_end = until or datetime.now(UTC)
        period_days = max((period_end - since).total_seconds() / 86400, 0.0001)

    report = {
        "total_releases_all_time": len(all_releases),
        "releases_in_window": len(windowed),
        "period_days": round(period_days, 2) if period_days is not None else None,
        "releases_per_week": round(len(windowed) / (period_days / 7), 4) if period_days else None,
        "tags": [
            {"tag": release["tag"], "date": release["date"].isoformat(), "subject": release["subject"]}
            for release in windowed
        ],
    }

    if args.json:
        print(json.dumps(report))
        return

    print("=== Deployment Frequency ===")
    print(f"total releases (all time): {report['total_releases_all_time']}")
    if period_days is not None:
        print(f"releases in window:       {report['releases_in_window']}  (window: {period_days:.1f} days)")
        if report["releases_per_week"] is not None:
            print(f"frequency:                 {report['releases_per_week']:.4f} releases/week")
    else:
        print(f"releases in window:       {report['releases_in_window']}  (no window given — showing all tags)")
    print()
    for release in windowed:
        subject = f"  - {release['subject'][:80]}" if release["subject"] else ""
        print(f"  {release['date'].isoformat()}  {release['tag']}{subject}")


if __name__ == "__main__":
    main()
