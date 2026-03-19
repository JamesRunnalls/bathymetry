#!/usr/bin/env python3
"""
swissBATHY3D Hillshade + Depth-Colored GeoTIFF Generator
=========================================================

Downloads bathymetry tiles from swisstopo's swissBATHY3D dataset via the
STAC API, mosaics them, computes a hillshade blended with a depth color
ramp, and writes a georeferenced TIFF reprojected to EPSG:4326.

Data source: https://www.swisstopo.admin.ch/en/height-model-swissbathy3d
Coordinate system: LV95 (EPSG:2056) → reprojected to WGS84 (EPSG:4326)

Usage:
    python swissbathy_hillshade.py --lake "Lake Zürich"
    python swissbathy_hillshade.py --lake "Lake Zürich" --azimuth 315 --altitude 45
    python swissbathy_hillshade.py --input-dir ./my_tiles --lake-level 406.0
    python swissbathy_hillshade.py                          # process all lakes
    python swissbathy_hillshade.py --help

Requirements:
    pip install rasterio numpy requests matplotlib cmocean
"""

import argparse
import glob
import io
import json
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path

# Project root is one level above this script (src/../)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_TILES_DIR = _PROJECT_ROOT / "data" / "tiles"
_DATA_OUTPUT_DIR = _PROJECT_ROOT / "data" / "output"

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling as WarpResampling,
)

try:
    import cmocean

    HAS_CMOCEAN = True
except ImportError:
    HAS_CMOCEAN = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants & lake metadata
# ─────────────────────────────────────────────────────────────────────────────

STAC_API_BASE = "https://data.geo.admin.ch/api/stac/v1"
COLLECTION_ID = "ch.swisstopo.swissbathy3d"
SRC_CRS = CRS.from_epsg(2056)  # LV95
DST_CRS = CRS.from_epsg(4326)  # WGS84

# Mapbox Terrain-RGB encoding constants
# Formula: height = -10000 + ((R * 256 * 256 + G * 256 + B) * 0.1)
TERRAIN_RGB_OFFSET  = 10000.0    # meters added before encoding
TERRAIN_RGB_SCALE   = 0.1        # meters per encoded unit (0.1 m resolution)
TERRAIN_RGB_MAX_VAL = 16_777_215  # 2^24 - 1 (max 3-byte value)

# Approximate lake surface elevations (m above sea level, LN02 datum).
# Keys match the STAC item ID slugs (swissbathy3d_<slug>).
# These are used to compute depth = surface_level - elevation.
LAKE_SURFACE_LEVELS = {
    "aegerisee":          724.0,
    "baldeggersee":       463.0,
    "bielersee":          429.1,
    "bodensee":           395.6,
    "brienzersee":        563.7,
    "hallwilersee":       449.3,
    "lacdejoux":         1004.0,
    "lacleman":           372.0,
    "lacneuchatel":       429.4,
    "lagomaggiore":       193.5,
    "lungernsee":         688.0,
    "murtensee":          429.3,
    "sarnersee":          468.4,
    "sempachersee":       504.0,
    "silsersee":         1797.0,
    "silvaplanersee":    1791.0,
    "thunersee":          557.8,
    "vierwaldstaettersee": 433.6,
    "walensee":           419.0,
    "zuerichsee":         406.0,
    "zugersee":           413.6,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data acquisition via STAC API
# ─────────────────────────────────────────────────────────────────────────────


def fetch_tile_urls(lake_name: str | None = None, bbox_lv95: tuple | None = None) -> list[dict]:
    """
    Query the swisstopo STAC API for swissBATHY3D items.

    Returns a list of dicts: [{"url": ..., "tile_id": ...}, ...]
    Each URL points to an ESRI ASCII Grid zip file.

    Parameters
    ----------
    lake_name : str, optional
        Filter by lake name (matched in item properties).
    bbox_lv95 : tuple, optional
        Bounding box in LV95 coords: (min_e, min_n, max_e, max_n).
    """
    import requests

    items = []
    url = f"{STAC_API_BASE}/collections/{COLLECTION_ID}/items?limit=100"

    if bbox_lv95:
        # STAC bbox is in WGS84, but swisstopo STAC accepts LV95 bbox
        url += f"&bbox={','.join(str(v) for v in bbox_lv95)}"

    page = 0
    while url:
        page += 1
        log.info(f"Fetching STAC items page {page}: {url}")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        for feature in data.get("features", []):
            props = feature.get("properties", {})
            assets = feature.get("assets", {})

            # Find the ESRI ASCII Grid asset
            for asset_key, asset in assets.items():
                href = asset.get("href", "")
                if "ESRIASCIIGRID" in href.upper() or "esriasciigrid" in href.lower():
                    item_id = feature.get("id", asset_key)

                    items.append({"url": href, "tile_id": item_id})

        # Follow pagination
        next_link = None
        for link in data.get("links", []):
            if link.get("rel") == "next":
                next_link = link.get("href")
                break
        url = next_link

    log.info(f"Found {len(items)} ESRI ASCII Grid tiles")

    if lake_name:
        # IDs are like "swissbathy3d_brienzersee" — match against the slug after the prefix
        def _slug(s):
            return s.lower().replace(" ", "").replace("-", "").replace("_", "")

        query = _slug(lake_name)
        filtered = [it for it in items if query in _slug(it["tile_id"])]
        if not filtered:
            available = sorted(it["tile_id"].replace("swissbathy3d_", "") for it in items)
            raise ValueError(
                f"No tiles found for '{lake_name}'. Available lakes:\n  " + "\n  ".join(available)
            )
        log.info(f"Filtered to {len(filtered)} tile(s) for '{lake_name}'")
        return filtered

    return items


def download_tiles(tile_urls: list[dict], output_dir: str) -> list[str]:
    """
    Download and extract ESRI ASCII Grid zip files.
    Returns list of paths to .asc files.
    """
    import requests

    asc_files = []
    os.makedirs(output_dir, exist_ok=True)

    for i, tile in enumerate(tile_urls):
        url = tile["url"]
        tile_id = tile["tile_id"]

        # Check if all files for this tile are already cached
        cached = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.lower().endswith((".asc", ".grd")) and tile_id in f
        ]
        if cached:
            log.info(f"Tile {i + 1}/{len(tile_urls)} already cached: {tile_id}")
            asc_files.extend(cached)
            continue

        log.info(f"Downloading tile {i + 1}/{len(tile_urls)}: {tile_id}")

        resp = requests.get(url, timeout=120)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for name in zf.namelist():
                if name.lower().endswith((".asc", ".grd")):
                    zf.extract(name, output_dir)
                    asc_files.append(os.path.join(output_dir, name))
                    log.info(f"  Extracted: {name}")

    log.info(f"Total .asc files extracted: {len(asc_files)}")
    return asc_files


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load & mosaic tiles
# ─────────────────────────────────────────────────────────────────────────────


def load_and_mosaic(asc_files: list[str], nodata: float = -9999.0) -> tuple[np.ndarray, rasterio.Affine, CRS]:
    """
    Open all .asc tiles, assign CRS EPSG:2056, and mosaic them.

    Returns
    -------
    elevation : np.ndarray (float32)
        Merged elevation grid with NaN for nodata.
    transform : rasterio.Affine
        Affine transform of the mosaic.
    crs : CRS
        Coordinate reference system (EPSG:2056).
    """
    datasets = []
    for f in asc_files:
        ds = rasterio.open(f)
        # ESRI ASCII grids from swissBATHY3D don't include a .prj — set CRS
        if ds.crs is None:
            # Re-open with CRS override
            ds.close()
            ds = rasterio.open(f, crs=SRC_CRS)
        datasets.append(ds)

    if not datasets:
        raise ValueError("No .asc files to mosaic")

    log.info(f"Mosaicking {len(datasets)} tiles...")
    mosaic_arr, mosaic_transform = merge(datasets, nodata=nodata)

    # Close all datasets
    for ds in datasets:
        ds.close()

    # Convert to float32 and replace nodata with NaN
    elevation = mosaic_arr[0].astype(np.float32)
    elevation[elevation == nodata] = np.nan
    # Also handle any other common nodata sentinels
    elevation[elevation < -9000] = np.nan

    log.info(f"Mosaic shape: {elevation.shape}, "
             f"elevation range: {np.nanmin(elevation):.1f} to {np.nanmax(elevation):.1f} m")

    return elevation, mosaic_transform, SRC_CRS


def load_or_mosaic_cached(
    asc_files: list[str],
    cache_path: str,
    nodata: float = -9999.0,
) -> tuple[np.ndarray, rasterio.Affine, CRS]:
    """
    Load a cached mosaic GeoTIFF if it is up-to-date, otherwise mosaic the
    source .asc files and write the result to cache_path for future runs.

    The cache is considered valid when cache_path exists and its modification
    time is >= the newest source .asc file.  Skipping ASCII parsing on reruns
    gives a significant speed-up for large lakes with many tiles.
    """
    cache = Path(cache_path)

    if asc_files and cache.exists():
        newest_src = max(os.path.getmtime(f) for f in asc_files)
        if cache.stat().st_mtime >= newest_src:
            log.info(f"Loading cached mosaic: {cache_path}")
            with rasterio.open(cache_path) as ds:
                elevation = ds.read(1).astype(np.float32)
                transform = ds.transform
            return elevation, transform, SRC_CRS

    elevation, transform, crs = load_and_mosaic(asc_files, nodata)

    cache.parent.mkdir(parents=True, exist_ok=True)
    h, w = elevation.shape
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": w,
        "height": h,
        "count": 1,
        "crs": SRC_CRS,
        "transform": transform,
        "compress": "lzw",
        "nodata": float("nan"),
    }
    with rasterio.open(cache_path, "w", **profile) as ds:
        ds.write(elevation, 1)
    log.info(f"Saved mosaic cache: {cache_path}")

    return elevation, transform, crs


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hillshade computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_hillshade(
    elevation: np.ndarray,
    transform: rasterio.Affine,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    multidirectional: bool = False,
    shadow_exponent: float = 2.0,
) -> np.ndarray:
    """
    Compute hillshade from an elevation array.

    Uses the standard Lambertian reflectance model:
        hillshade = cos(zenith)*cos(slope) + sin(zenith)*sin(slope)*cos(azimuth - aspect)

    Parameters
    ----------
    elevation : np.ndarray
        Elevation grid (NaN for nodata).
    transform : rasterio.Affine
        Affine transform (used to get cell size).
    azimuth : float
        Sun azimuth in degrees (0=N, 90=E, 180=S, 270=W).
    altitude : float
        Sun altitude angle in degrees above horizon.
    z_factor : float
        Vertical exaggeration factor.
    multidirectional : bool
        If True, average hillshades from 4 directions for uniform illumination.

    Returns
    -------
    hillshade : np.ndarray (float32, 0–1)
    """
    cellsize_x = abs(transform.a)
    cellsize_y = abs(transform.e)

    # Apply z-factor
    elev = elevation * z_factor

    # Pad edges to compute gradients cleanly
    elev_padded = np.pad(elev, 1, mode="edge")

    # Gradient using Horn's method (3x3 kernel)
    dz_dx = (
        (elev_padded[0:-2, 2:] + 2 * elev_padded[1:-1, 2:] + elev_padded[2:, 2:])
        - (elev_padded[0:-2, :-2] + 2 * elev_padded[1:-1, :-2] + elev_padded[2:, :-2])
    ) / (8.0 * cellsize_x)

    dz_dy = (
        (elev_padded[2:, 0:-2] + 2 * elev_padded[2:, 1:-1] + elev_padded[2:, 2:])
        - (elev_padded[0:-2, 0:-2] + 2 * elev_padded[0:-2, 1:-1] + elev_padded[0:-2, 2:])
    ) / (8.0 * cellsize_y)

    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect = np.arctan2(-dz_dy, dz_dx)

    def _shade(az_deg, alt_deg):
        az_rad = np.deg2rad(360.0 - az_deg + 90.0)  # Convert to math convention
        zen_rad = np.deg2rad(90.0 - alt_deg)
        hs = np.cos(zen_rad) * np.cos(slope) + np.sin(zen_rad) * np.sin(slope) * np.cos(az_rad - aspect)
        return np.clip(hs, 0, 1)

    if multidirectional:
        # Average from 4 azimuths for more uniform illumination
        azimuths = [0, 90, 180, 270]
        weights = [1, 1, 1, 1]
        hs = np.zeros_like(elevation)
        for az, w in zip(azimuths, weights):
            hs += w * _shade(az, altitude)
        hs /= sum(weights)
    else:
        hs = _shade(azimuth, altitude)

    # Center hillshade so a flat horizontal surface = 0.5 (neutral).
    # Without this, a flat lake floor gets hs ≈ cos(zenith) ≈ 0.707 (bright),
    # which after multiply-blending makes the deep flat floor appear lighter
    # than shadow-facing shallow slopes — visually inverting deep vs. shallow.
    floor_hs = np.cos(np.deg2rad(90.0 - altitude))
    hs = hs - floor_hs + 0.5
    hs = np.clip(hs, 0, 1)

    # Shadow exponent: darken only the shadow half (hs < 0.5).
    # Neutral (0.5) and highlights (> 0.5) are left unchanged.
    if shadow_exponent != 1.0:
        shadow_mask = hs < 0.5
        hs[shadow_mask] = 0.5 * np.power(hs[shadow_mask] / 0.5, shadow_exponent)

    # Where elevation is NaN, hillshade is NaN too
    hs[np.isnan(elevation)] = np.nan

    log.info(f"Hillshade computed (azimuth={azimuth}°, altitude={altitude}°, z_factor={z_factor})")
    return hs.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Depth colorization
# ─────────────────────────────────────────────────────────────────────────────


def make_depth_colormap(name: str = "auto", cmap_min: float = 0.0, cmap_max: float = 1.0) -> mcolors.Colormap:
    """
    Create or load a bathymetric depth colormap.

    Parameters
    ----------
    name : str
        'auto' uses the custom blue ramp. Any matplotlib colormap name also works.
    cmap_min, cmap_max : float
        Subset of the colormap to use (0–1). E.g. cmap_min=0.25 skips the first 25%.
    """
    if name == "auto":
        # Ramp: index 0 = deepest (#12222F dark navy), index 1 = shallowest (#7d9dac teal)
        colors = ["#12222F", "#7d9dac"]
        cmap = mcolors.LinearSegmentedColormap.from_list("bathy_custom", colors, N=256)
    else:
        cmap = plt.get_cmap(name)

    if cmap_min != 0.0 or cmap_max != 1.0:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"{cmap.name}_sub",
            cmap(np.linspace(cmap_min, cmap_max, 256)),
            N=256,
        )
    return cmap


def colorize_depth(
    elevation: np.ndarray,
    lake_surface_level: float,
    cmap: mcolors.Colormap | None = None,
    depth_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Map elevation to RGBA colors based on depth below lake surface.

    Parameters
    ----------
    elevation : np.ndarray
        Elevation grid (m above sea level).
    lake_surface_level : float
        Lake surface elevation in meters.
    cmap : Colormap
        Matplotlib colormap to use.
    depth_range : tuple, optional
        (min_depth, max_depth) for normalization. Auto-detected if None.

    Returns
    -------
    rgba : np.ndarray (H, W, 4) uint8
    """
    if cmap is None:
        cmap = make_depth_colormap()

    depth = elevation - lake_surface_level  # negative = deeper (0 at surface)
    depth[depth > 0] = 0  # clip above-surface points to surface level

    if depth_range is None:
        dmin = float(np.nanmin(depth))  # most negative = deepest
        dmax = 0.0                      # surface
    else:
        dmin, dmax = depth_range

    if dmax <= dmin:
        dmin = dmax - 1.0

    log.info(f"Depth range: {abs(dmin):.1f} m max depth (lake surface = {lake_surface_level:.1f} m)")

    # Normalize: 0 = deepest, 1 = surface/shallow
    norm_depth = (depth - dmin) / (dmax - dmin)
    norm_depth = np.clip(norm_depth, 0, 1)
    norm_depth = np.sqrt(norm_depth)  # gamma: more color contrast near surface

    # Apply colormap
    rgba = cmap(norm_depth)  # returns float 0–1 RGBA
    rgba = (rgba * 255).astype(np.uint8)

    # Set alpha=0 for NaN (nodata) pixels — outside lake is fully transparent
    nan_mask = np.isnan(elevation)
    rgba[nan_mask] = [0, 0, 0, 0]

    log.info(f"Depth colorization complete")
    return rgba


# ─────────────────────────────────────────────────────────────────────────────
# 5. Composite hillshade + color
# ─────────────────────────────────────────────────────────────────────────────


def composite_hillshade_color(
    rgba: np.ndarray,
    hillshade: np.ndarray,
    blend_mode: str = "multiply",
    hillshade_strength: float = 0.6,
) -> np.ndarray:
    """
    Blend hillshade shading into depth-colored RGBA image.

    Parameters
    ----------
    rgba : np.ndarray (H, W, 4) uint8
        Depth-colored image.
    hillshade : np.ndarray (H, W) float32 0–1
        Hillshade intensity.
    blend_mode : str
        'multiply' or 'overlay'.
    hillshade_strength : float (0–1)
        How much the hillshade affects the color. 0 = no shading, 1 = full.

    Returns
    -------
    result : np.ndarray (H, W, 4) uint8
    """
    result = rgba.copy().astype(np.float32)
    hs = hillshade.copy()
    hs[np.isnan(hs)] = 0

    if blend_mode == "multiply":
        # Lerp between unshaded color and multiply-blended color
        for c in range(3):  # R, G, B
            original = result[:, :, c]
            shaded = original * hs
            result[:, :, c] = original * (1 - hillshade_strength) + shaded * hillshade_strength

    elif blend_mode == "overlay":
        # Overlay blend: combines multiply and screen
        for c in range(3):
            base = result[:, :, c] / 255.0
            overlay = hs
            blended = np.where(
                base < 0.5,
                2 * base * overlay * 255.0,
                (1 - 2 * (1 - base) * (1 - overlay)) * 255.0,
            )
            result[:, :, c] = result[:, :, c] * (1 - hillshade_strength) + blended * hillshade_strength

    result[:, :, :3] = np.clip(result[:, :, :3], 0, 255)
    # Preserve alpha from original
    result[:, :, 3] = rgba[:, :, 3]

    log.info(f"Composite complete (mode={blend_mode}, strength={hillshade_strength})")
    return result.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Write GeoTIFF with reprojection
# ─────────────────────────────────────────────────────────────────────────────


def write_geotiff(
    rgba: np.ndarray,
    src_transform: rasterio.Affine,
    src_crs: CRS,
    output_path: str,
    dst_crs: CRS = DST_CRS,
    resampling: WarpResampling = WarpResampling.bilinear,
) -> str:
    """
    Write the RGBA composite as a georeferenced TIFF, reprojected to dst_crs.

    Parameters
    ----------
    rgba : np.ndarray (H, W, 4) uint8
    src_transform : Affine
    src_crs : CRS (source, typically EPSG:2056)
    output_path : str
    dst_crs : CRS (destination, default EPSG:4326)
    resampling : Resampling method

    Returns
    -------
    output_path : str
    """
    h, w = rgba.shape[:2]
    bands = rgba.shape[2]  # 4 for RGBA

    # Compute the destination transform and dimensions
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, w, h, *rasterio.transform.array_bounds(h, w, src_transform)
    )

    log.info(f"Reprojecting {src_crs} → {dst_crs}: {w}x{h} → {dst_width}x{dst_height}")

    # Prepare destination array
    dst_data = np.zeros((bands, dst_height, dst_width), dtype=np.uint8)

    # Reproject each band
    for band_idx in range(bands):
        src_band = rgba[:, :, band_idx]
        reproject(
            source=src_band,
            destination=dst_data[band_idx],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )

    # Write to GeoTIFF
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": dst_width,
        "height": dst_height,
        "count": bands,
        "crs": dst_crs,
        "transform": dst_transform,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "photometric": "rgb",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        for band_idx in range(bands):
            dst.write(dst_data[band_idx], band_idx + 1)

    # Mark band 4 as alpha so GIS tools recognise transparency automatically
    with rasterio.open(output_path, "r+") as dst:
        dst.colorinterp = (
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
            rasterio.enums.ColorInterp.alpha,
        )
        dst.build_overviews([2, 4, 8, 16], Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    log.info(f"GeoTIFF written: {output_path} ({file_size:.1f} MB)")
    log.info(f"  CRS: {dst_crs}")
    log.info(f"  Dimensions: {dst_width} x {dst_height}")
    log.info(f"  Bounds: {rasterio.transform.array_bounds(dst_height, dst_width, dst_transform)}")

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 6b. Terrain-RGB encoding
# ─────────────────────────────────────────────────────────────────────────────


def encode_terrain_rgb(
    elevation: np.ndarray,
    nodata_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode a float32 elevation array into Mapbox Terrain-RGB channels.

    Encoding formula (Mapbox spec):
        height = -10000 + ((R * 65536 + G * 256 + B) * 0.1)

    Parameters
    ----------
    elevation : np.ndarray (H, W) float32
        Elevation values in metres (any datum).
    nodata_mask : np.ndarray (H, W) bool, optional
        True where pixels have no data.  If None, derived from NaN values.

    Returns
    -------
    R, G, B, alpha : four np.ndarray (H, W) uint8
        alpha = 255 for valid pixels, 0 for nodata.
    """
    if nodata_mask is None:
        nodata_mask = np.isnan(elevation)
    valid = ~nodata_mask & ~np.isnan(elevation)

    R     = np.zeros(elevation.shape, dtype=np.uint8)
    G     = np.zeros(elevation.shape, dtype=np.uint8)
    B     = np.zeros(elevation.shape, dtype=np.uint8)
    alpha = np.zeros(elevation.shape, dtype=np.uint8)

    if not np.any(valid):
        log.warning("encode_terrain_rgb: no valid pixels in elevation array")
        return R, G, B, alpha

    # Clamp to spec range [-10000, 1677711.5] before conversion
    max_height = TERRAIN_RGB_MAX_VAL * TERRAIN_RGB_SCALE - TERRAIN_RGB_OFFSET
    elev_clamped = np.where(valid, np.clip(elevation, -TERRAIN_RGB_OFFSET, max_height), 0.0)

    # Convert to integer units using int64 to avoid any overflow
    value = np.where(
        valid,
        np.round((elev_clamped + TERRAIN_RGB_OFFSET) / TERRAIN_RGB_SCALE).astype(np.int64),
        0,
    ).astype(np.int64)
    value = np.clip(value, 0, TERRAIN_RGB_MAX_VAL)

    R[valid] = ((value[valid] >> 16) & 0xFF).astype(np.uint8)
    G[valid] = ((value[valid] >>  8) & 0xFF).astype(np.uint8)
    B[valid] = ( value[valid]        & 0xFF).astype(np.uint8)
    alpha[valid] = 255

    log.info(
        f"Terrain-RGB: {np.count_nonzero(valid)} valid pixels, "
        f"value range {np.nanmin(elevation[valid]):.1f}–{np.nanmax(elevation[valid]):.1f} m"
    )
    return R, G, B, alpha


def write_terrain_rgb_geotiff(
    elevation: np.ndarray,
    src_transform: rasterio.Affine,
    src_crs: CRS,
    output_path: str,
    lake_surface_level: float,
    dst_crs: CRS = DST_CRS,
    nodata_mask: np.ndarray | None = None,
) -> str:
    """
    Write a Mapbox Terrain-RGB encoded GeoTIFF encoding water depth.

    Depth is computed as ``lake_surface_level - elevation`` so that 0 m is
    the lake surface and values increase with depth.  The output is a 4-band
    uint8 GeoTIFF (R, G, B, alpha) reprojected to dst_crs.  Nearest-neighbour
    resampling is used throughout — bilinear or average resampling would
    corrupt the encoded values.

    Decode client-side with:
        depth = -10000 + ((R * 65536 + G * 256 + B) * 0.1)

    Parameters
    ----------
    elevation : np.ndarray (H, W) float32
    src_transform : Affine
    src_crs : CRS (typically EPSG:2056)
    output_path : str
    lake_surface_level : float
        Lake surface elevation (m, LN02).  Used to convert elevation to depth.
    dst_crs : CRS (default EPSG:4326)
    nodata_mask : np.ndarray (H, W) bool, optional

    Returns
    -------
    output_path : str
    """
    # Convert absolute elevation to depth below the lake surface
    depth = np.where(np.isnan(elevation), np.nan, lake_surface_level - elevation)
    log.info(
        f"Terrain-RGB: lake surface {lake_surface_level} m, "
        f"depth range 0–{np.nanmax(depth):.1f} m"
    )
    R, G, B, alpha = encode_terrain_rgb(depth, nodata_mask=nodata_mask)

    # Stack into (H, W, 4) to match write_geotiff's layout convention
    rgba = np.stack([R, G, B, alpha], axis=-1)

    h, w = rgba.shape[:2]
    bands = 4

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, w, h, *rasterio.transform.array_bounds(h, w, src_transform)
    )

    log.info(f"Terrain-RGB: reprojecting {src_crs} → {dst_crs}: {w}x{h} → {dst_width}x{dst_height}")

    dst_data = np.zeros((bands, dst_height, dst_width), dtype=np.uint8)

    for band_idx in range(bands):
        reproject(
            source=rgba[:, :, band_idx],
            destination=dst_data[band_idx],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=WarpResampling.nearest,  # must not interpolate encoded RGB values
        )

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": dst_width,
        "height": dst_height,
        "count": bands,
        "crs": dst_crs,
        "transform": dst_transform,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        for band_idx in range(bands):
            dst.write(dst_data[band_idx], band_idx + 1)

    with rasterio.open(output_path, "r+") as dst:
        dst.colorinterp = (
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
            rasterio.enums.ColorInterp.alpha,
        )
        # Use nearest resampling for overviews — average would corrupt encoded values
        dst.build_overviews([2, 4, 8, 16], Resampling.nearest)
        dst.update_tags(ns="rio_overview", resampling="nearest")

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    log.info(f"Terrain-RGB GeoTIFF written: {output_path} ({file_size:.1f} MB)")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 7. Legend generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_legend(
    cmap: mcolors.Colormap,
    depth_min: float,
    depth_max: float,
    output_path: str,
    lake_name: str = "",
) -> str:
    """Generate a colorbar legend image showing the depth-to-color mapping."""
    fig, ax = plt.subplots(figsize=(6, 1.2))
    norm = mcolors.Normalize(vmin=depth_min, vmax=depth_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cb.set_label(f"Depth (m){' — ' + lake_name if lake_name else ''}", fontsize=10)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    log.info(f"Legend saved: {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 8. Convenience: load from local directory of .asc files
# ─────────────────────────────────────────────────────────────────────────────


def find_asc_files(directory: str) -> list[str]:
    """Recursively find all .asc and .grd files in a directory."""
    patterns = ["**/*.asc", "**/*.grd", "**/*.ASC", "**/*.GRD"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(directory, pattern), recursive=True))
    return sorted(set(files))


# ─────────────────────────────────────────────────────────────────────────────
# 9. Synthetic demo data (for testing without network access)
# ─────────────────────────────────────────────────────────────────────────────


def generate_demo_data(output_dir: str) -> list[str]:
    """
    Generate a small set of synthetic .asc tiles mimicking a lake basin.
    Useful for testing the full pipeline without downloading real data.

    Creates a ~3x3 km simulated lakebed centered around LV95 coordinates
    that resemble a plausible Swiss lake location.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = []

    # Simulate tiles around E=2683, N=1248 (roughly Lake Zürich area)
    base_e, base_n = 2683000, 1248000
    cellsize = 2  # 2m resolution for demo
    tile_size = 500  # 500 cells = 1km at 2m

    for de in range(3):
        for dn in range(3):
            e0 = base_e + de * tile_size * cellsize
            n0 = base_n + dn * tile_size * cellsize

            # Create a bowl-shaped lakebed
            y_coords = np.arange(tile_size) * cellsize + n0
            x_coords = np.arange(tile_size) * cellsize + e0
            xx, yy = np.meshgrid(x_coords, y_coords)

            # Center of the lake
            cx = base_e + 1.5 * tile_size * cellsize
            cy = base_n + 1.5 * tile_size * cellsize

            # Distance from center (normalized)
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            max_radius = 1.4 * tile_size * cellsize

            # Bowl shape: deeper at center, shallower at edges
            lake_surface = 406.0  # Lake Zürich approximate
            max_depth = 136.0  # Lake Zürich max depth

            # Parabolic basin with some noise
            np.random.seed(de * 10 + dn)
            relative_dist = dist / max_radius
            depth = max_depth * (1 - relative_dist**2)
            depth = np.clip(depth, 0, max_depth)

            # Add terrain noise
            noise = np.random.normal(0, 2, depth.shape)
            # Smooth noise
            from scipy.ndimage import gaussian_filter
            noise = gaussian_filter(noise, sigma=5)
            depth += noise
            depth = np.clip(depth, 0, max_depth)

            elevation = lake_surface - depth

            # Mask outside the lake (circular boundary with fuzzy edge)
            outside = relative_dist > 1.0
            elevation[outside] = -9999.0

            # Write ESRI ASCII Grid
            fname = f"swissBATHY3D_{int(e0 / 1000)}_{int(n0 / 1000)}.asc"
            fpath = os.path.join(output_dir, fname)

            with open(fpath, "w") as f:
                f.write(f"ncols {tile_size}\n")
                f.write(f"nrows {tile_size}\n")
                f.write(f"xllcorner {e0}\n")
                f.write(f"yllcorner {n0}\n")
                f.write(f"cellsize {cellsize}\n")
                f.write(f"NODATA_value -9999\n")
                # Write rows top-to-bottom
                for row in range(tile_size - 1, -1, -1):
                    vals = " ".join(f"{v:.2f}" for v in elevation[row])
                    f.write(vals + "\n")

            files.append(fpath)
            log.info(f"  Generated demo tile: {fname}")

    return files


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────


def run_pipeline(
    asc_files: list[str],
    output_path: str,
    lake_surface_level: float,
    lake_name: str = "",
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 3.0,
    multidirectional: bool = False,
    blend_mode: str = "multiply",
    hillshade_strength: float = 0.8,
    shadow_exponent: float = 2.0,
    colormap_name: str = "auto",
    cmap_min: float = 0.0,
    cmap_max: float = 1.0,
    depth_range: tuple[float, float] | None = None,
    dst_crs: CRS = DST_CRS,
    legend: bool = True,
    terrain_rgb: bool = False,
    cache_path: str | None = None,
) -> str:
    """Run the full processing pipeline."""

    log.info("=" * 60)
    log.info("swissBATHY3D Hillshade + Depth Color Pipeline")
    log.info("=" * 60)

    # Step 1: Mosaic (or load from cache)
    log.info("\n── Step 1: Loading and mosaicking tiles ──")
    if cache_path:
        elevation, transform, crs = load_or_mosaic_cached(asc_files, cache_path)
    else:
        elevation, transform, crs = load_and_mosaic(asc_files)

    # Step 1b: Terrain-RGB (optional — uses raw elevation before any visualisation)
    if terrain_rgb:
        log.info("\n── Step 1b: Writing Terrain-RGB GeoTIFF ──")
        terrain_rgb_path = output_path.replace("_hillshade_4326", "_terrain_rgb_4326")
        if terrain_rgb_path == output_path:
            base, ext = os.path.splitext(output_path)
            terrain_rgb_path = base + "_terrain_rgb_4326" + (ext or ".tif")
        write_terrain_rgb_geotiff(elevation, transform, crs, terrain_rgb_path,
                                   lake_surface_level=lake_surface_level, dst_crs=dst_crs)

    # Step 2: Hillshade
    log.info("\n── Step 2: Computing hillshade ──")
    hs = compute_hillshade(
        elevation, transform,
        azimuth=azimuth, altitude=altitude,
        z_factor=z_factor, multidirectional=multidirectional,
        shadow_exponent=shadow_exponent,
    )

    # Step 3: Depth colorization
    log.info("\n── Step 3: Colorizing depth ──")
    cmap = make_depth_colormap(colormap_name, cmap_min=cmap_min, cmap_max=cmap_max)
    rgba = colorize_depth(elevation, lake_surface_level, cmap=cmap, depth_range=depth_range)

    # Step 4: Composite
    log.info("\n── Step 4: Compositing hillshade + color ──")
    composited = composite_hillshade_color(rgba, hs, blend_mode=blend_mode, hillshade_strength=hillshade_strength)

    # Step 5: Write GeoTIFF
    log.info("\n── Step 5: Writing georeferenced TIFF ──")
    write_geotiff(composited, transform, crs, output_path, dst_crs=dst_crs)

    # Step 6: Optional legend
    if legend:
        legend_path = output_path.replace(".tif", "_legend.png").replace(".tiff", "_legend.png")
        dmax = float(np.nanmax(lake_surface_level - elevation[~np.isnan(elevation)]))
        generate_legend(cmap, 0, dmax, legend_path, lake_name=lake_name)

    log.info("\n── Done! ──")
    log.info(f"Output: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate hillshade + depth-colored GeoTIFF from swissBATHY3D data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download & process a lake via STAC API:
  python swissbathy_hillshade.py --lake "Lake Zürich"

  # Process locally downloaded .asc tiles:
  python swissbathy_hillshade.py --input-dir ./tiles --lake-level 406.0

  # Run with demo/synthetic data:
  python swissbathy_hillshade.py --demo

  # Custom hillshade settings:
  python swissbathy_hillshade.py --demo --azimuth 270 --altitude 30 --z-factor 2.0 --multidirectional
        """,
    )

    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--lake", type=str, help="Lake name to download from STAC API (e.g. 'Lake Zürich'). Omit to process all lakes.")
    source.add_argument("--input-dir", type=str, help="Directory with pre-downloaded .asc tiles")
    source.add_argument("--demo", action="store_true", help="Use synthetic demo data for testing")

    parser.add_argument("--lake-level", type=float, default=None,
                        help="Lake surface elevation (m). Auto-detected if --lake is used.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output GeoTIFF path (default: auto-named)")
    parser.add_argument("--azimuth", type=float, default=315.0,
                        help="Sun azimuth in degrees (default: 315)")
    parser.add_argument("--altitude", type=float, default=35.0,
                        help="Sun altitude in degrees (default: 45)")
    parser.add_argument("--z-factor", type=float, default=3.0,
                        help="Vertical exaggeration (default: 3.0)")
    parser.add_argument("--multidirectional", action="store_true",
                        help="Use 4-direction averaged hillshade")
    parser.add_argument("--blend-mode", choices=["multiply", "overlay"], default="multiply",
                        help="How to blend hillshade with color (default: multiply)")
    parser.add_argument("--hillshade-strength", type=float, default=0.6,
                        help="Hillshade blending strength 0-1 (default: 0.6)")
    parser.add_argument("--shadow-exponent", type=float, default=4.0,
                        help="Gamma applied to hillshade; >1 = darker shadows (default: 2.0)")
    parser.add_argument("--colormap", type=str, default="auto",
                        help="Colormap name (default: auto = cmocean.deep or custom blue)")
    parser.add_argument("--colormap-range", type=float, nargs=2, default=[0.0, 1.0],
                        metavar=("MIN", "MAX"),
                        help="Subset of colormap to use, e.g. --colormap-range 0.25 1.0 skips the first 25%%")
    parser.add_argument("--no-legend", action="store_true", help="Skip legend generation")
    parser.add_argument(
        "--terrain-rgb",
        action="store_true",
        help="Also output a Mapbox Terrain-RGB encoded GeoTIFF ({lake}_terrain_rgb_4326.tif). "
             "Decode elevation with: height = -10000 + ((R*65536 + G*256 + B) * 0.1)",
    )
    parser.add_argument("--dst-crs", type=str, default="EPSG:4326",
                        help="Output CRS (default: EPSG:4326)")

    args = parser.parse_args()

    # Resolve lake surface level
    def _slug(s):
        return s.lower().replace(" ", "").replace("-", "").replace("_", "").replace("ü", "ue").replace("ä", "ae").replace("ö", "oe").replace("â", "a").replace("ê", "e").replace("î", "i").replace("ô", "o").replace("û", "u")

    lake_name = ""
    if args.lake:
        lake_name = args.lake
        if args.lake_level is None:
            query = _slug(args.lake)
            # Exact slug match first, then partial
            match = next((k for k in LAKE_SURFACE_LEVELS if _slug(k) == query), None)
            if match is None:
                match = next((k for k in LAKE_SURFACE_LEVELS if query in _slug(k) or _slug(k) in query), None)
            if match:
                lake_level = LAKE_SURFACE_LEVELS[match]
                lake_name = match
            else:
                parser.error(
                    f"Lake surface level not found for '{args.lake}'. "
                    f"Use --lake-level to specify it manually.\n"
                    f"Known lakes: {', '.join(LAKE_SURFACE_LEVELS.keys())}"
                )
        else:
            lake_level = args.lake_level
    elif args.demo:
        lake_level = 406.0
        lake_name = "Demo Lake (synthetic)"
    elif args.input_dir:
        if args.lake_level is None:
            parser.error("--lake-level is required when using --input-dir")
        lake_level = args.lake_level
    else:
        # No source specified — will process all lakes; lake_level unused at top level
        lake_level = 0.0

    dst_crs = CRS.from_user_input(args.dst_crs)

    def _run_one(lake_slug: str, lake_surface: float):
        log.info(f"=== Processing {lake_slug} (surface {lake_surface} m) ===")
        tile_urls = fetch_tile_urls(lake_name=lake_slug)
        if not tile_urls:
            log.warning(f"No tiles found for '{lake_slug}', skipping.")
            return
        dl_dir = str(_DATA_TILES_DIR / f"swissbathy_tiles_{lake_slug}")
        asc_files = download_tiles(tile_urls, dl_dir)
        out = args.output if args.output else str(_DATA_OUTPUT_DIR / f"{lake_slug}_hillshade_4326.tif")
        cache_path = str(_DATA_TILES_DIR / f"{lake_slug}_mosaic.tif")
        run_pipeline(
            asc_files=asc_files,
            output_path=out,
            cache_path=cache_path,
            lake_surface_level=lake_surface,
            lake_name=lake_slug,
            azimuth=args.azimuth,
            altitude=args.altitude,
            z_factor=args.z_factor,
            multidirectional=args.multidirectional,
            blend_mode=args.blend_mode,
            hillshade_strength=args.hillshade_strength,
            shadow_exponent=args.shadow_exponent,
            colormap_name=args.colormap,
            cmap_min=args.colormap_range[0],
            cmap_max=args.colormap_range[1],
            dst_crs=dst_crs,
            legend=not args.no_legend,
            terrain_rgb=args.terrain_rgb,
        )

    # Get .asc files
    if args.demo:
        demo_dir = os.path.join(tempfile.gettempdir(), "swissbathy_demo")
        log.info("Generating synthetic demo data...")
        asc_files = generate_demo_data(demo_dir)
        output_path = args.output or str(_DATA_OUTPUT_DIR / "demo_hillshade_4326.tif")
        run_pipeline(
            asc_files=asc_files,
            output_path=output_path,
            lake_surface_level=lake_level,
            lake_name=lake_name,
            azimuth=args.azimuth,
            altitude=args.altitude,
            z_factor=args.z_factor,
            multidirectional=args.multidirectional,
            blend_mode=args.blend_mode,
            hillshade_strength=args.hillshade_strength,
            shadow_exponent=args.shadow_exponent,
            colormap_name=args.colormap,
            cmap_min=args.colormap_range[0],
            cmap_max=args.colormap_range[1],
            dst_crs=dst_crs,
            legend=not args.no_legend,
            terrain_rgb=args.terrain_rgb,
        )
    elif args.input_dir:
        asc_files = find_asc_files(args.input_dir)
        if not asc_files:
            parser.error(f"No .asc/.grd files found in {args.input_dir}")
        log.info(f"Found {len(asc_files)} tiles in {args.input_dir}")
        safe_name = lake_name.lower().replace(" ", "_") if lake_name else "swissbathy"
        output_path = args.output or str(_DATA_OUTPUT_DIR / f"{safe_name}_hillshade_4326.tif")
        run_pipeline(
            asc_files=asc_files,
            output_path=output_path,
            lake_surface_level=lake_level,
            lake_name=lake_name,
            azimuth=args.azimuth,
            altitude=args.altitude,
            z_factor=args.z_factor,
            multidirectional=args.multidirectional,
            blend_mode=args.blend_mode,
            hillshade_strength=args.hillshade_strength,
            shadow_exponent=args.shadow_exponent,
            colormap_name=args.colormap,
            cmap_min=args.colormap_range[0],
            cmap_max=args.colormap_range[1],
            dst_crs=dst_crs,
            legend=not args.no_legend,
            terrain_rgb=args.terrain_rgb,
        )
    elif args.lake:
        _run_one(lake_name, lake_level)
    else:
        # No lake specified — run for all known lakes
        log.info(f"No lake specified, processing all {len(LAKE_SURFACE_LEVELS)} lakes...")
        for slug, surface in LAKE_SURFACE_LEVELS.items():
            _run_one(slug, surface)


if __name__ == "__main__":
    main()
