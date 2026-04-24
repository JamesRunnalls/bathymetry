#!/usr/bin/env python3
"""
Glacier Ice Thickness Terrain-RGB GeoTIFF Generator
====================================================

Encodes Swiss Alps glacier ice thickness (IceThickness.tif) into Mapbox
Terrain-RGB format, reprojected to EPSG:4326.

Data source: DOI 10.3929/ethz-b-000434697 (CC BY 4.0)
Input CRS: LV95 (EPSG:2056) → reprojected to WGS84 (EPSG:4326)

Decode client-side with:
    thickness_m = -10000 + ((R * 65536 + G * 256 + B) * 0.1)

Usage:
    python glacier_terrain_rgb.py
    python glacier_terrain_rgb.py --input /path/to/IceThickness.tif --output /path/to/out.tif
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, Resampling as WarpResampling

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swissbathy_hillshade import encode_terrain_rgb, SRC_CRS, DST_CRS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = Path("/home/runnalja/git/headwater/external/04_IceThickness_SwissAlps/IceThickness.tif")
DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "output" / "glacier_ice_thickness_terrain_rgb.tif"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def run(input_path: Path, output_path: Path) -> None:
    log.info(f"Reading {input_path}")
    with rasterio.open(input_path) as src:
        thickness = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs

    nodata_mask = (thickness == -9999) | (thickness < 0)
    thickness[nodata_mask] = np.nan
    log.info(f"Valid pixels: {np.count_nonzero(~nodata_mask):,}  "
             f"thickness range: 0–{np.nanmax(thickness):.1f} m")

    R, G, B, alpha = encode_terrain_rgb(thickness, nodata_mask=nodata_mask)
    rgba = np.stack([R, G, B, alpha], axis=-1)

    h, w = rgba.shape[:2]
    dst_transform, dst_width, dst_height = calculate_default_transform(
        crs, DST_CRS, w, h, *rasterio.transform.array_bounds(h, w, transform)
    )
    log.info(f"Reprojecting {crs} → {DST_CRS}: {w}x{h} → {dst_width}x{dst_height}")

    dst_data = np.zeros((4, dst_height, dst_width), dtype=np.uint8)
    for band_idx in range(4):
        reproject(
            source=rgba[:, :, band_idx],
            destination=dst_data[band_idx],
            src_transform=transform,
            src_crs=crs,
            dst_transform=dst_transform,
            dst_crs=DST_CRS,
            resampling=WarpResampling.nearest,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": dst_width,
        "height": dst_height,
        "count": 4,
        "crs": DST_CRS,
        "transform": dst_transform,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        for band_idx in range(4):
            dst.write(dst_data[band_idx], band_idx + 1)

    with rasterio.open(output_path, "r+") as dst:
        dst.colorinterp = (
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
            rasterio.enums.ColorInterp.alpha,
        )
        dst.build_overviews([2, 4, 8, 16], Resampling.nearest)
        dst.update_tags(ns="rio_overview", resampling="nearest")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log.info(f"Written: {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Encode glacier ice thickness as Terrain-RGB GeoTIFF")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input IceThickness.tif")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output GeoTIFF path")
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
