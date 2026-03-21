# swissBATHY3D Hillshade Pipeline

Generates hillshade + depth-colored GeoTIFFs and XYZ map tiles from
[swissBATHY3D](https://www.swisstopo.admin.ch/en/height-model-swissbathy3d)
bathymetry data via the swisstopo STAC API.

## Repository structure

```
bathymetry/
├── src/
│   ├── swissbathy_hillshade.py   # Download tiles, compute hillshade, write GeoTIFFs
│   └── generate_tiles.py         # Slice GeoTIFFs into XYZ PNG tiles (gdal2tiles)
├── docs/
│   └── example_commands.txt      # Useful one-liners
├── data/                         # gitignored — generated/downloaded files
│   ├── tiles/                    # Raw .asc tiles downloaded from swisstopo STAC
│   ├── output/                   # GeoTIFFs and legend PNGs
│   └── xyz/                      # XYZ PNG tile directories
├── .gitignore
└── README.md
```

## Requirements

```bash
conda install -c conda-forge rasterio numpy requests matplotlib cmocean gdal
```

`rclone` must be installed and configured separately if using `--upload`.

## Usage

### 1. Generate GeoTIFFs

```bash
# Single lake (downloads tiles automatically)
python src/swissbathy_hillshade.py --lake "zugersee"

# All lakes
python src/swissbathy_hillshade.py

# With custom hillshade settings
python src/swissbathy_hillshade.py --multidirectional --colormap Blues_r --colormap-range 0.0 0.5 --terrain-rgb

# From pre-downloaded .asc tiles
python src/swissbathy_hillshade.py --input-dir data/tiles/swissbathy_tiles_zugersee \
    --lake-level 413.6

# Synthetic demo (no network required)
python src/swissbathy_hillshade.py --demo
```

Output GeoTIFFs are written to `data/output/`.

### 2. Generate XYZ map tiles

```bash
# Tile all GeoTIFFs in a folder
python src/generate_tiles.py data/output/

# Custom zoom range
python src/generate_tiles.py data/output/ --zoom 7-14

# Generate and upload to an rclone remote
python src/generate_tiles.py data/output/ --upload cloudflare_r2_personal:rivers
```

Each `.tif` in `data/output/` produces a tile directory in `data/xyz/`:

```
data/output/ageri_hillshade_4326.tif   →  data/xyz/tiles_ageri_hillshade/
data/output/biel_terrain_rgb_4326.tif  →  data/xyz/tiles_biel_terrain/
```

Tile generation is skipped if the output directory already exists.
With `--upload`, each tile set is copied to `REMOTE:BUCKET/tiles_<name>` via `rclone copy` (already-uploaded files are skipped).

### Terrain-RGB depth decoding

The `--terrain-rgb` flag produces a Mapbox Terrain-RGB encoded GeoTIFF where depth
(metres below lake surface) is encoded in the RGB channels:

```javascript
const depth = -10000 + (R * 65536 + G * 256 + B) * 0.1;
```

## Supported lakes

Aegerisee, Baldeggersee, Bielersee, Bodensee, Brienzersee, Hallwilersee,
Lac de Joux, Lac Léman, Lac de Neuchâtel, Lago Maggiore, Lungernsee,
Murtensee, Sarnersee, Sempachersee, Silsersee, Silvaplanersee, Thunersee,
Vierwaldstättersee, Walensee, Zürichsee, Zugersee.

## Data source

swisstopo swissBATHY3D — © swisstopo
STAC API: `https://data.geo.admin.ch/api/stac/v1/collections/ch.swisstopo.swissbathy3d`
