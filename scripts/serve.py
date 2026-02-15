# /// script
# requires-python = ">=3.10"
# dependencies = ["RangeHTTPServer"]
# ///
"""CORS + Range HTTP server for viewing zarr stores in neuroglancer.

Usage:
    uv run scripts/serve.py <zarr_path> [--port PORT] [--array NAME] [--no-open]

Opens neuroglancer in your browser pointing at the zarr array.

Examples:
    uv run scripts/serve.py build/visual.zarr
    uv run scripts/serve.py build/visual.zarr --array 0 --port 9000
"""

import argparse
import json
import os
import webbrowser
from http.server import HTTPServer
from pathlib import Path
from urllib.parse import quote

from RangeHTTPServer import RangeRequestHandler


class Handler(RangeRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS, HEAD")
        self.send_header(
            "Access-Control-Expose-Headers", "Content-Range, Content-Length"
        )
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


def neuroglancer_url(port, zarr_rel_path):
    source = f"zarr://http://localhost:{port}/{zarr_rel_path}"
    state = json.dumps(
        {"layers": [{"type": "image", "source": source, "name": "data"}]},
        separators=(",", ":"),
    )
    return f"https://neuroglancer-demo.appspot.com/#!{quote(state)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", help="Path to zarr store")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument(
        "--array", default="0", help="Array name within the store (default: 0)"
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Don't open browser automatically"
    )
    args = parser.parse_args()

    zarr_path = Path(args.zarr_path).resolve()
    serve_dir = zarr_path.parent
    zarr_rel = f"{zarr_path.name}/{args.array}"

    os.chdir(serve_dir)

    url = neuroglancer_url(args.port, zarr_rel)
    print(f"Serving {serve_dir} on http://localhost:{args.port}")
    print(f"Neuroglancer: {url}")
    print("Press Ctrl+C to stop.")

    if not args.no_open:
        webbrowser.open(url)

    server = HTTPServer(("", args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
