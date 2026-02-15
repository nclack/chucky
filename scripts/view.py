# /// script
# requires-python = ">=3.10"
# dependencies = ["napari[all]", "zarr"]
# ///
"""Open a zarr store in napari.

Usage:
    uv run scripts/view.py <path> [--array NAME] [--contrast MIN MAX]

Examples:
    uv run scripts/view.py build/visual.zarr
    uv run scripts/view.py build/visual.zarr --array 0 --contrast 0 16000
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to zarr store")
    parser.add_argument(
        "--array", default="0", help="Array name within the store (default: 0)"
    )
    parser.add_argument(
        "--contrast",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Contrast limits (default: auto)",
    )
    args = parser.parse_args()

    import napari
    import zarr

    array_path = os.path.join(args.path, args.array)
    store = zarr.open(array_path, mode="r")
    print(f"shape: {store.shape}  dtype: {store.dtype}")

    viewer = napari.Viewer()
    kwargs = {"name": os.path.basename(args.path)}
    if args.contrast:
        kwargs["contrast_limits"] = args.contrast
    viewer.add_image(store, **kwargs)
    napari.run()


if __name__ == "__main__":
    main()
