#!/usr/bin/env python3
"""
Bathymetry DEM Generator — Swiss Lakes (Vector25)
==================================================
Creates a bathymetry DEM for each lake in data/vector25/bathymetry_Lakes_CH.shp.
Contour lines are grouped by NAME, interpolated, and saved as per-lake GeoTIFFs.

Z values: negative depth below lake surface (Depth field, negated).

Usage:
  # All lakes
  python bathymetry_dem.py

  # Single lake
  python bathymetry_dem.py --lake Bodensee

  # Custom resolution / method
  python bathymetry_dem.py --resolution 25 --method linear
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

DEFAULT_SHP = Path(__file__).parent.parent / "data" / "vector25" / "bathymetry_Lakes_CH.shp"
DEFAULT_OUT = Path(__file__).parent.parent / "data" / "dem"
DEFAULT_HILLSHADE_OUT = Path(__file__).parent.parent / "data" / "vector" / "output"


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------

def list_lakes(shp_path):
    """Return sorted list of unique lake names in the shapefile."""
    import fiona
    names = set()
    with fiona.open(shp_path) as src:
        for feat in src:
            name = feat["properties"].get("NAME")
            if name:
                names.add(name)
    return sorted(names)


def load_lake_contours(shp_path, lake_name, sample_spacing=None):
    """Load bathymetry contour points for one lake.

    Returns (points, crs) where points is Nx3 array with z = -Depth (negative below surface).
    """
    import fiona
    from shapely.geometry import shape

    points = []
    crs = None
    with fiona.open(shp_path) as src:
        crs = src.crs
        for feat in src:
            if feat["properties"].get("NAME") != lake_name:
                continue
            depth = feat["properties"].get("Depth")
            if depth is None:
                continue
            z = -float(depth)  # negate: 0 = surface, negative = deeper
            geom = shape(feat["geometry"])
            coords = _extract_coords(geom, z, sample_spacing)
            points.extend(coords)

    if not points:
        raise ValueError(f"No features found for lake: {lake_name!r}")

    return np.array(points), crs


def _extract_coords(geom, z_default, sample_spacing):
    """Extract (x, y, z) from a LineString / MultiLineString geometry."""
    from shapely.geometry import LineString, MultiLineString

    coords = []
    lines = []

    if isinstance(geom, LineString):
        lines = [geom]
    elif isinstance(geom, MultiLineString):
        lines = list(geom.geoms)

    for line in lines:
        if sample_spacing and line.length > sample_spacing:
            distances = np.arange(0, line.length, sample_spacing)
            distances = np.append(distances, line.length)
            for d in distances:
                pt = line.interpolate(d)
                c = pt.coords[0]
                coords.append((c[0], c[1], z_default))
        else:
            for c in line.coords:
                coords.append((c[0], c[1], z_default))

    return coords


# ---------------------------------------------------------------------------
# 2. INTERPOLATION
# ---------------------------------------------------------------------------

def interpolate_linear(points, grid_x, grid_y):
    from scipy.interpolate import griddata
    return griddata(points[:, :2], points[:, 2], (grid_x, grid_y), method="linear")


def interpolate_cubic(points, grid_x, grid_y):
    from scipy.interpolate import griddata
    return griddata(points[:, :2], points[:, 2], (grid_x, grid_y), method="cubic")


def interpolate_rbf(points, grid_x, grid_y, smoothing=0.0):
    from scipy.interpolate import RBFInterpolator
    interp = RBFInterpolator(points[:, :2], points[:, 2],
                             kernel="thin_plate_spline", smoothing=smoothing)
    flat = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    return interp(flat).reshape(grid_x.shape)


def interpolate_idw(points, grid_x, grid_y, power=2, n_neighbors=12):
    from scipy.spatial import cKDTree
    tree = cKDTree(points[:, :2])
    flat = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    dists, idxs = tree.query(flat, k=n_neighbors)
    dists = np.maximum(dists, 1e-10)
    weights = 1.0 / dists ** power
    z_vals = points[idxs, 2]
    return (np.sum(weights * z_vals, axis=1) / np.sum(weights, axis=1)).reshape(grid_x.shape)


METHODS = {
    "linear": interpolate_linear,
    "cubic": interpolate_cubic,
    "rbf": interpolate_rbf,
    "idw": interpolate_idw,
}


# ---------------------------------------------------------------------------
# 3. GRID & OUTPUT
# ---------------------------------------------------------------------------

def build_grid(points, resolution, padding_pct=0.02):
    xmin, ymin = points[:, :2].min(axis=0)
    xmax, ymax = points[:, :2].max(axis=0)
    dx = (xmax - xmin) * padding_pct
    dy = (ymax - ymin) * padding_pct
    xmin -= dx; xmax += dx
    ymin -= dy; ymax += dy
    xi = np.arange(xmin, xmax, resolution)
    yi = np.arange(ymin, ymax, resolution)
    grid_x, grid_y = np.meshgrid(xi, yi)
    return grid_x, grid_y, (xmin, xmax, ymin, ymax)


def mask_extrapolation(grid_z, points, grid_x, grid_y, max_dist):
    from scipy.spatial import cKDTree
    tree = cKDTree(points[:, :2])
    flat = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    dists, _ = tree.query(flat, k=1)
    grid_z[dists.reshape(grid_x.shape) > max_dist] = np.nan
    return grid_z


def save_geotiff(grid_z, extent, resolution, output_path, crs=None, nodata=-9999):
    import rasterio
    from rasterio.transform import from_bounds

    xmin, xmax, ymin, ymax = extent
    nrows, ncols = grid_z.shape
    grid_out = np.flipud(np.where(np.isnan(grid_z), nodata, grid_z))
    transform = from_bounds(xmin, ymin, xmax, ymax, ncols, nrows)

    crs_out = None
    if crs:
        if isinstance(crs, dict):
            crs_out = rasterio.crs.CRS.from_dict(crs)
        elif isinstance(crs, str):
            crs_out = rasterio.crs.CRS.from_string(crs)
        else:
            crs_out = crs

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", driver="GTiff",
                       height=nrows, width=ncols, count=1,
                       dtype="float32", crs=crs_out, transform=transform,
                       nodata=nodata, compress="deflate") as dst:
        dst.write(grid_out.astype(np.float32), 1)

    print(f"  [saved] {output_path}  ({ncols}x{nrows}, z: {np.nanmin(grid_z):.1f} to {np.nanmax(grid_z):.1f} m)")


def safe_filename(name):
    return re.sub(r"[^\w\-]", "_", name)


def _rasterio_crs(fiona_crs):
    import rasterio
    if fiona_crs is None:
        return None
    if isinstance(fiona_crs, dict):
        return rasterio.crs.CRS.from_dict(fiona_crs)
    if isinstance(fiona_crs, str):
        return rasterio.crs.CRS.from_string(fiona_crs)
    return fiona_crs  # already a rasterio CRS or compatible object


def save_hillshade(grid_z, extent, resolution, crs, lake_name, out_dir,
                   azimuth=315.0, altitude=35.0, z_factor=3.0,
                   hillshade_strength=0.6, shadow_exponent=4.0,
                   colormap="Blues_r", colormap_range=(0.0, 0.5),
                   multidirectional=False):
    """Generate a hillshade + depth-colored RGBA GeoTIFF in EPSG:4326,
    matching the swissbathy_hillshade output format."""
    import sys
    import rasterio
    from rasterio.transform import from_bounds
    sys.path.insert(0, str(Path(__file__).parent))
    from swissbathy_hillshade import (
        compute_hillshade, make_depth_colormap,
        colorize_depth, composite_hillshade_color, write_geotiff,
    )

    nrows, ncols = grid_z.shape
    xmin, xmax, ymin, ymax = extent
    transform = from_bounds(xmin, ymin, xmax, ymax, ncols, nrows)
    src_crs = _rasterio_crs(crs)

    # compute_hillshade expects raster convention: row 0 = north (top)
    grid_raster = np.flipud(grid_z.copy())
    grid_raster[grid_raster >= 0] = np.nan  # surface pixels → transparent

    hs = compute_hillshade(grid_raster, transform,
                           azimuth=azimuth, altitude=altitude,
                           z_factor=z_factor, shadow_exponent=shadow_exponent,
                           multidirectional=multidirectional)

    cmap = make_depth_colormap(colormap, cmap_min=colormap_range[0], cmap_max=colormap_range[1])
    # z is already relative (0=surface, negative=deeper), so lake_surface_level=0
    rgba = colorize_depth(grid_raster, lake_surface_level=0.0, cmap=cmap)
    composited = composite_hillshade_color(rgba, hs, hillshade_strength=hillshade_strength)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / (safe_filename(lake_name) + "_hillshade_4326.tif"))
    write_geotiff(composited, transform, src_crs, out_path)
    print(f"  [hillshade] {out_path}")


# ---------------------------------------------------------------------------
# 4. PER-LAKE PIPELINE
# ---------------------------------------------------------------------------

def process_lake(lake_name, shp_path, out_dir, resolution=25, method="linear",
                 sample_spacing=None, mask_distance=None, smoothing=0.0,
                 hillshade=False, hillshade_out_dir=None,
                 colormap="Blues_r", colormap_range=(0.0, 0.5), multidirectional=False):
    print(f"\n{'─'*50}")
    print(f"  {lake_name}")
    print(f"{'─'*50}")

    try:
        points, crs = load_lake_contours(shp_path, lake_name, sample_spacing)
    except ValueError as e:
        print(f"  [SKIP] {e}")
        return

    # Clamp Z to [surface, -9999] to drop any bad shoreline duplicates noise
    points = points[points[:, 2] <= 0]
    if len(points) < 4:
        print(f"  [SKIP] Too few points after filtering ({len(points)})")
        return

    print(f"  Points: {len(points)}  |  depth range: {points[:,2].min():.1f} to {points[:,2].max():.1f} m")

    grid_x, grid_y, extent = build_grid(points, resolution)
    print(f"  Grid:   {grid_x.shape[1]} x {grid_x.shape[0]} @ {resolution} m")

    if method == "rbf":
        grid_z = interpolate_rbf(points, grid_x, grid_y, smoothing=smoothing)
    else:
        grid_z = METHODS[method](points, grid_x, grid_y)

    if mask_distance:
        grid_z = mask_extrapolation(grid_z, points, grid_x, grid_y, mask_distance)

    # Clamp interpolated values to valid range
    grid_z = np.clip(grid_z, points[:, 2].min(), 0)

    fname = safe_filename(lake_name) + "_bathymetry.tif"
    save_geotiff(grid_z, extent, resolution, str(out_dir / fname), crs)

    if hillshade:
        save_hillshade(grid_z, extent, resolution, crs, lake_name,
                       hillshade_out_dir or DEFAULT_HILLSHADE_OUT,
                       colormap=colormap, colormap_range=colormap_range,
                       multidirectional=multidirectional)


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate bathymetry DEMs for Swiss lakes from Vector25 shapefile")
    parser.add_argument("--shp", default=str(DEFAULT_SHP), help="Input shapefile")
    parser.add_argument("-o", "--output-dir", default=str(DEFAULT_OUT), help="Output directory")
    parser.add_argument("--lake", help="Process a single lake by name (default: all)")
    parser.add_argument("-r", "--resolution", type=float, default=25, help="Grid cell size in metres")
    parser.add_argument("-m", "--method", default="linear", choices=list(METHODS.keys()))
    parser.add_argument("--sample-spacing", type=float, help="Densify contour lines at this interval")
    parser.add_argument("--mask-distance", type=float, help="Mask cells beyond this distance from data")
    parser.add_argument("--smoothing", type=float, default=0.0, help="RBF smoothing factor")
    parser.add_argument("--hillshade", action="store_true",
                        help="Also output a hillshade RGBA GeoTIFF (EPSG:4326) matching swissbathy format")
    parser.add_argument("--hillshade-output-dir", default=str(DEFAULT_HILLSHADE_OUT),
                        help=f"Directory for hillshade outputs (default: {DEFAULT_HILLSHADE_OUT})")
    parser.add_argument("--colormap", default="Blues_r",
                        help="Matplotlib colormap name for depth colouring (default: Blues_r)")
    parser.add_argument("--colormap-range", type=float, nargs=2, default=[0.0, 0.5],
                        metavar=("MIN", "MAX"),
                        help="Subset of colormap to use (default: 0.0 0.5)")
    parser.add_argument("--multidirectional", action="store_true",
                        help="Use 4-direction averaged hillshade")
    parser.add_argument("--list-lakes", action="store_true", help="Print available lake names and exit")
    args = parser.parse_args()

    shp_path = Path(args.shp)
    if not shp_path.exists():
        print(f"[ERROR] Shapefile not found: {shp_path}")
        sys.exit(1)

    if args.list_lakes:
        for name in list_lakes(shp_path):
            print(name)
        return

    out_dir = Path(args.output_dir)

    if args.lake:
        lakes = [args.lake]
    else:
        lakes = list_lakes(shp_path)
        print(f"Processing {len(lakes)} lakes → {out_dir}")

    for lake in lakes:
        process_lake(
            lake_name=lake,
            shp_path=shp_path,
            out_dir=out_dir,
            resolution=args.resolution,
            method=args.method,
            sample_spacing=args.sample_spacing,
            mask_distance=args.mask_distance,
            smoothing=args.smoothing,
            hillshade=args.hillshade,
            hillshade_out_dir=Path(args.hillshade_output_dir),
            colormap=args.colormap,
            colormap_range=tuple(args.colormap_range),
            multidirectional=args.multidirectional,
        )

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
