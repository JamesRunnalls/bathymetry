#!/usr/bin/env python3
"""
Generate XYZ PNG tiles from bathymetry GeoTIFFs using gdal2tiles.py.

Auto mode (default) — scans a directory for all hillshade GeoTIFFs and pairs
each one with its Terrain-RGB counterpart if present:

    python generate_tiles.py --zoom 10-13

    python generate_tiles.py --dir /path/to/tiffs --zoom 10-13

Manual mode — process a single explicit pair:

    python generate_tiles.py \\
        --visual  zugersee_hillshade_4326.tif \\
        --terrain zugersee_terrain_rgb_4326.tif \\
        --output-dir zugersee_tiles \\
        --zoom 10-13

Output structure (per lake)
---------------------------
    {slug}_tiles/
        visual/{z}/{x}/{y}.png    RGBA hillshade tiles (bilinear resampled)
        terrain/{z}/{x}/{y}.png   Terrain-RGB depth tiles (nearest resampled)
        tilejson.json             bounds, zoom range, tile URL templates

Leaflet example
---------------
    L.tileLayer('zugersee_tiles/visual/{z}/{x}/{y}.png', {
        minZoom: 10, maxZoom: 13
    }).addTo(map);

Decode depth in JavaScript from a terrain tile pixel (R, G, B):
    const depth = -10000 + (R * 65536 + G * 256 + B) * 0.1;
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import rasterio

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_OUTPUT_DIR = _PROJECT_ROOT / "data" / "output"


def find_gdal2tiles() -> str:
    """Locate the gdal2tiles.py executable on PATH."""
    for name in ("gdal2tiles.py", "gdal2tiles"):
        path = shutil.which(name)
        if path:
            return path
    sys.exit(
        "Error: gdal2tiles.py not found.\n"
        "Ensure GDAL is installed and active (e.g. `conda activate bathy`)."
    )


def find_tiff_pairs(directory: str) -> list[tuple[str, str, str | None]]:
    """
    Scan *directory* for hillshade GeoTIFFs and pair each with its
    Terrain-RGB counterpart if one exists.

    Returns
    -------
    list of (slug, visual_path, terrain_path_or_None)
        slug is the lake name derived from the filename, e.g. 'zugersee'.
    """
    pattern = os.path.join(directory, "*_hillshade_4326.tif")
    visual_tifs = sorted(glob.glob(pattern))
    if not visual_tifs:
        sys.exit(
            f"No *_hillshade_4326.tif files found in '{directory}'.\n"
            "Run swissbathy_hillshade.py first, or specify --visual explicitly."
        )

    pairs = []
    for visual in visual_tifs:
        slug = os.path.basename(visual).replace("_hillshade_4326.tif", "")
        terrain = visual.replace("_hillshade_4326.tif", "_terrain_rgb_4326.tif")
        terrain = terrain if os.path.isfile(terrain) else None
        pairs.append((slug, visual, terrain))
    return pairs


def run_gdal2tiles(
    gdal2tiles_path: str,
    input_tif: str,
    output_dir: str,
    zoom_range: tuple[int, int],
    resampling: str,
    processes: int = 4,
) -> None:
    """
    Call gdal2tiles.py via subprocess.

    Use resampling='near' for Terrain-RGB tiles — bilinear resampling corrupts
    the RGB-encoded depth values at byte boundaries.
    Use resampling='bilinear' for visual/hillshade tiles.
    """
    cmd = (
        [sys.executable, gdal2tiles_path]
        if gdal2tiles_path.endswith(".py")
        else [gdal2tiles_path]
    )
    cmd += [
        "--xyz",                              # XYZ slippy-map numbering (Leaflet default)
        f"--zoom={zoom_range[0]}-{zoom_range[1]}",
        f"--resampling={resampling}",
        f"--processes={processes}",
        "--exclude",                          # skip fully-transparent tiles
        "--webviewer=none",                   # no auto-generated HTML viewer
        input_tif,
        output_dir,
    ]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def count_tiles(directory: str) -> dict[int, int]:
    """Return {zoom_level: tile_count} by walking the tile directory tree."""
    counts: dict[int, int] = {}
    if not os.path.isdir(directory):
        return counts
    for z_name in os.listdir(directory):
        z_path = os.path.join(directory, z_name)
        if not os.path.isdir(z_path) or not z_name.isdigit():
            continue
        z = int(z_name)
        n = sum(len(files) for _, _, files in os.walk(z_path))
        if n:
            counts[z] = n
    return counts


def write_tilejson(
    output_dir: str,
    bounds: tuple[float, float, float, float],
    zoom_range: tuple[int, int],
    has_terrain: bool,
) -> str:
    """Write a TileJSON 2.2.0 metadata file to output_dir/tilejson.json."""
    west, south, east, north = bounds
    data: dict = {
        "tilejson": "2.2.0",
        "minzoom": zoom_range[0],
        "maxzoom": zoom_range[1],
        "bounds": [west, south, east, north],
        "visual_tiles": "visual/{z}/{x}/{y}.png",
    }
    if has_terrain:
        data["terrain_tiles"] = "terrain/{z}/{x}/{y}.png"

    path = os.path.join(output_dir, "tilejson.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def process_pair(
    gdal2tiles_path: str,
    slug: str,
    visual: str,
    terrain: str | None,
    output_dir: str,
    zoom_range: tuple[int, int],
    processes: int,
) -> None:
    """Generate tiles and tilejson.json for one visual+terrain pair."""
    print(f"\n{'═' * 60}")
    print(f"  {slug}")
    print(f"{'═' * 60}")
    print(f"  visual  : {visual}")
    print(f"  terrain : {terrain or '(none)'}")
    print(f"  output  : {os.path.abspath(output_dir)}")

    # Visual tiles
    visual_dir = os.path.join(output_dir, "visual")
    print(f"\n── Visual tiles → {visual_dir}")
    run_gdal2tiles(gdal2tiles_path, visual, visual_dir, zoom_range,
                   resampling="bilinear", processes=processes)

    # Terrain-RGB tiles
    terrain_dir = None
    if terrain:
        terrain_dir = os.path.join(output_dir, "terrain")
        print(f"\n── Terrain-RGB tiles → {terrain_dir}")
        print("   (nearest-neighbour resampling preserves RGB-encoded depth values)")
        run_gdal2tiles(gdal2tiles_path, terrain, terrain_dir, zoom_range,
                       resampling="near", processes=processes)

    # tilejson.json
    with rasterio.open(visual) as src:
        b = src.bounds  # GeoTIFFs are in EPSG:4326
        bounds = (b.left, b.bottom, b.right, b.top)

    tilejson_path = write_tilejson(output_dir, bounds, zoom_range,
                                   has_terrain=terrain is not None)

    # Per-lake summary
    print(f"\n── {slug} summary ──")
    visual_counts = count_tiles(visual_dir)
    for z in sorted(visual_counts):
        print(f"  visual  zoom {z:2d}: {visual_counts[z]:4d} tiles")
    if terrain_dir:
        terrain_counts = count_tiles(terrain_dir)
        for z in sorted(terrain_counts):
            print(f"  terrain zoom {z:2d}: {terrain_counts[z]:4d} tiles")
    print(f"  tilejson: {tilejson_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate XYZ PNG tiles from bathymetry GeoTIFFs using gdal2tiles.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto mode: tile all lakes in the current directory
  python generate_tiles.py --zoom 10-13

  # Auto mode: scan a specific directory
  python generate_tiles.py --dir /data/bathymetry --zoom 10-13

  # Manual mode: single explicit pair
  python generate_tiles.py \\
      --visual zugersee_hillshade_4326.tif \\
      --terrain zugersee_terrain_rgb_4326.tif \\
      --output-dir zugersee_tiles --zoom 10-13
        """,
    )

    # Auto-discovery options
    parser.add_argument(
        "--dir", default=str(_DATA_OUTPUT_DIR), metavar="DIR",
        help="Directory to scan for *_hillshade_4326.tif files (default: data/output/). "
             "Ignored when --visual is provided.",
    )

    # Manual single-pair options
    parser.add_argument(
        "--visual", default=None, metavar="GEOTIFF",
        help="Hillshade RGBA GeoTIFF. Providing this enables manual mode.",
    )
    parser.add_argument(
        "--terrain", default=None, metavar="GEOTIFF",
        help="Terrain-RGB depth GeoTIFF (optional, manual mode only).",
    )
    parser.add_argument(
        "--output-dir", default=None, metavar="DIR",
        help="Output directory (manual mode only). "
             "In auto mode each lake gets its own {slug}_tiles/ directory.",
    )

    # Shared options
    parser.add_argument(
        "--zoom", default="10-14", metavar="MIN-MAX",
        help="Zoom level range, e.g. '10-14' (default: 10-14)",
    )
    parser.add_argument(
        "--processes", type=int, default=4, metavar="N",
        help="Parallel gdal2tiles worker processes (default: 4)",
    )

    args = parser.parse_args()

    # Parse zoom range
    try:
        z_parts = args.zoom.split("-")
        zoom_range = (int(z_parts[0]), int(z_parts[1]))
    except (ValueError, IndexError):
        parser.error(f"--zoom must be MIN-MAX, e.g. '10-14', got '{args.zoom}'")
        return

    gdal2tiles_path = find_gdal2tiles()

    if args.visual:
        # ── Manual mode: single explicit pair ───────────────────────────────
        if not os.path.isfile(args.visual):
            parser.error(f"--visual: file not found: {args.visual}")
        if args.terrain and not os.path.isfile(args.terrain):
            parser.error(f"--terrain: file not found: {args.terrain}")
        if args.output_dir is None:
            parser.error("--output-dir is required in manual mode")

        slug = os.path.basename(args.visual).replace("_hillshade_4326.tif", "")
        process_pair(gdal2tiles_path, slug, args.visual, args.terrain,
                     args.output_dir, zoom_range, args.processes)
    else:
        # ── Auto mode: discover all pairs in --dir ───────────────────────────
        directory = os.path.abspath(args.dir)
        pairs = find_tiff_pairs(directory)

        print(f"Found {len(pairs)} hillshade GeoTIFF(s) in '{directory}':")
        for slug, visual, terrain in pairs:
            terrain_label = os.path.basename(terrain) if terrain else "(no terrain-RGB)"
            print(f"  {slug}: {os.path.basename(visual)}  +  {terrain_label}")

        for slug, visual, terrain in pairs:
            output_dir = os.path.join(directory, f"{slug}_tiles")  # sibling of TIFFs in data/output/
            process_pair(gdal2tiles_path, slug, visual, terrain,
                         output_dir, zoom_range, args.processes)

    print(f"\n{'═' * 60}")
    print("  All done.")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
