# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pydantic",
# ]
# ///
"""
Generate a self-contained HTML+D3 report from benchmark sweep results.

Usage:
    uv run scripts/sweep/report.py results.json -o build/html/report.html
    uv run scripts/sweep/report.py bench/results/*.json
    uv run scripts/sweep/report.py --results-dir bench/results/

Note: The report embeds JSON data at generation time. Re-run this script
after modifying results files to see updated data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from models import migrate_results, validate_results

TEMPLATE_PATH = Path(__file__).parent / "template.html"


def load_files(paths: list[Path], *, warn: bool = True) -> list[dict]:
    """Load result files, returning a list of {label, machine, runs} dicts."""
    files = []
    for p in paths:
        with open(p) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping corrupt JSON file {p}: {e}", file=sys.stderr)
                continue

        migrate_results(data)

        if warn:
            try:
                validate_results(data)
            except ValidationError as e:
                for err in e.errors():
                    loc = " -> ".join(str(x) for x in err["loc"])
                    print(f"Warning: {p.name}: {loc}: {err['msg']}", file=sys.stderr)

        machine = data.get("machine", {})
        hostname = machine.get("hostname", "")
        commit = machine.get("commit", "unknown")
        date = machine.get("date", "")[:10]
        parts = [x for x in [hostname, commit, date] if x]
        label = " ".join(parts) if parts else "unknown"
        files.append({
            "label": label,
            "filename": p.name,
            "machine": machine,
            "runs": data.get("runs", []),
        })
    return files


def main():
    ap = argparse.ArgumentParser(description="Generate HTML report from sweep results")
    ap.add_argument("input", type=Path, nargs="*", help="Result JSON file(s) from sweep.py")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="Directory to glob for *.json result files")
    ap.add_argument("-o", "--output", type=Path, default=Path("build/html/report.html"),
                    help="Output HTML path")
    args = ap.parse_args()

    paths: list[Path] = list(args.input or [])
    if args.results_dir:
        paths.extend(sorted(args.results_dir.glob("*.json")))
    if not paths:
        ap.error("No input files. Provide paths or use --results-dir.")

    files = load_files(paths)
    total_runs = sum(len(f["runs"]) for f in files)
    print(f"Loaded {len(files)} file(s), {total_runs} runs", file=sys.stderr)

    data = {"version": 2, "files": files}

    template = TEMPLATE_PATH.read_text()
    json_str = json.dumps(data).replace("</", "<\\/")
    html = template.replace("__DATA_PLACEHOLDER__", json_str)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024:.0f} KiB)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
