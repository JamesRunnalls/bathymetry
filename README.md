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
│   └── output/                   # GeoTIFFs, legend PNGs, XYZ tile directories
├── .gitignore
└── README.md
```

## Requirements

```bash
conda install -c conda-forge rasterio numpy requests matplotlib cmocean
```

## Usage

### 1. Generate GeoTIFFs

```bash
# Single lake (downloads tiles automatically)
python src/swissbathy_hillshade.py --lake "zugersee"

# All lakes
python src/swissbathy_hillshade.py

# With custom hillshade settings
python src/swissbathy_hillshade.py --lake "zugersee" \
    --multidirectional --colormap Blues_r --colormap-range 0.0 0.75 --terrain-rgb

# From pre-downloaded .asc tiles
python src/swissbathy_hillshade.py --input-dir data/tiles/swissbathy_tiles_zugersee \
    --lake-level 413.6

# Synthetic demo (no network required)
python src/swissbathy_hillshade.py --demo
```

Output GeoTIFFs are written to `data/output/`.

### 2. Generate XYZ map tiles

```bash
# Tile all lakes in data/output/ (default)
python src/generate_tiles.py --zoom 10-14

# Single lake
python src/generate_tiles.py \
    --visual  data/output/zugersee_hillshade_4326.tif \
    --terrain data/output/zugersee_terrain_rgb_4326.tif \
    --output-dir data/output/zugersee_tiles \
    --zoom 10-14
```

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
