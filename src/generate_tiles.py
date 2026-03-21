#!/usr/bin/env python3
"""
Generate XYZ PNG tiles from GeoTIFFs using gdal2tiles.py.

Usage:
    python generate_tiles.py /path/to/tiffs
    python generate_tiles.py /path/to/tiffs --zoom 7-14 --processes 4
    python generate_tiles.py /path/to/tiffs --upload cloudflare_r2_personal:rivers

For each *.tif in the folder a tile set is created:
    ageri_hillshade_4326.tif   →  tiles_ageri_hillshade/
    biel_terrain_rgb_4326.tif  →  tiles_biel_terrain/

Skips tile generation if the output directory already exists.
With --upload, uploads each tile set to the given rclone remote/bucket prefix,
skipping any that are already present at the destination.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_gdal2tiles() -> str:
    env_bin = Path(sys.executable).parent
    for name in ("gdal2tiles.py", "gdal2tiles"):
        # Prefer the executable in the active Python env over any system install
        candidate = env_bin / name
        if candidate.is_file():
            return str(candidate)
        path = shutil.which(name)
        if path:
            return path
    sys.exit(
        "Error: gdal2tiles.py not found.\n"
        "Ensure GDAL is installed and active (e.g. `conda activate bathy`)."
    )


def output_dir_for(tif_path: str) -> str:
    stem = Path(tif_path).stem                      # e.g. biel_terrain_rgb_4326
    stem = stem.removesuffix("_rgb_4326")           # terrain: biel_terrain_rgb_4326 → biel_terrain
    stem = stem.removesuffix("_4326")               # others:  ageri_hillshade_4326  → ageri_hillshade
    xyz_dir = Path(tif_path).parent.parent / "xyz"
    return str(xyz_dir / f"tiles_{stem}")


def rclone_upload(local_dir: str, remote_prefix: str) -> None:
    """Upload a tile directory with rclone, skipping files already at the destination."""
    dir_name = os.path.basename(local_dir)
    destination = f"{remote_prefix}/{dir_name}"
    cmd = ["rclone", "copy", "--progress", local_dir, destination]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_gdal2tiles(gdal2tiles_path: str, tif: str, out_dir: str,
                   zoom: str, processes: int) -> None:
    env_bin = str(Path(sys.executable).parent)
    if gdal2tiles_path.endswith(".py") and gdal2tiles_path.startswith(env_bin):
        cmd = [sys.executable, gdal2tiles_path]
    else:
        cmd = [gdal2tiles_path]
    cmd += [
        "--xyz",
        f"--zoom={zoom}",
        f"--processes={processes}",
        "--exclude",
        "--webviewer=none",
        tif,
        out_dir,
    ]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir", metavar="DIR", help="Folder containing GeoTIFFs")
    parser.add_argument("--zoom", default="7-14", metavar="MIN-MAX",
                        help="Zoom range (default: 7-14)")
    parser.add_argument("--processes", type=int, default=4, metavar="N",
                        help="Parallel worker processes (default: 4)")
    parser.add_argument("--upload", metavar="REMOTE:BUCKET",
                        help="rclone remote and bucket prefix to upload each tile set to, "
                             "e.g. 'cloudflare_r2_personal:rivers'")
    args = parser.parse_args()

    folder = os.path.abspath(args.dir)
    tifs = sorted(glob.glob(os.path.join(folder, "*.tif")))
    if not tifs:
        sys.exit(f"No .tif files found in '{folder}'.")

    gdal2tiles_path = find_gdal2tiles()

    for tif in tifs:
        out_dir = output_dir_for(tif)
        name = os.path.basename(tif)
        if os.path.isdir(out_dir):
            print(f"  skip  {name}  (tiles already exist: {os.path.basename(out_dir)})")
        else:
            print(f"\n  {name}  →  {os.path.basename(out_dir)}")
            run_gdal2tiles(gdal2tiles_path, tif, out_dir, args.zoom, args.processes)

        if args.upload:
            print(f"\n  uploading {os.path.basename(out_dir)}  →  {args.upload}/")
            rclone_upload(out_dir, args.upload)

    print("\nDone.")


if __name__ == "__main__":
    main()
