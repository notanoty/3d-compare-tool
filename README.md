# Mesh Compare

Drop an original model and a new scan, get plain accuracy numbers (RMS, % within tolerance, etc.).

## Run

```sh
.venv/bin/mesh-compare
```

First-time setup (Python 3.11+):

```sh
python3.13 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Use

1. Drop the **original model** into the left box and the **new scan** into the right box
   (or click a box to browse). Supported: STL, PLY, OBJ, OFF, glTF.
2. Set the tolerance (default ±0.1 mm) and click **Compare**.
3. **Copy results** puts a tab-separated table on the clipboard for Excel / Google Sheets.

### Trimming

When a model is added, a trimming window opens (turn off with *Open trimming when a model
is added*, or use the **Trim…** buttons later). It shows the model from the front, right,
back and left.

- Drag on any view to draw the cut line; drag its round ends to tilt or move it.
- **Remove top / Remove bottom** picks which side is cut away (shown in red on every view).
- **Up axis** changes which way is "up" if the model is lying on its side.
- **Also apply this cut to …** aligns the other model and cuts it at the same place.
- **Trim & save** writes `<name>_trimmed.stl` next to the original file (the original is
  never changed). Comparisons then use the trimmed files.

### Matching scale

Phone/photo scans (e.g. Polycam photo mode) often have no real-world size. Click
**Match scale…** to see both models side by side at the same mm-per-pixel over a
millimetre grid, then scale the scan:

- drag the slider or type an exact factor; **× 1000 (m → mm)** and **× 10 (cm → mm)**
  fix plain unit mix-ups;
- **Auto-fit** estimates the factor by fitting the scan's shape to the original — check
  that the two then look the same size;
- **Apply** saves `<name>_scaled.stl` and uses it; **Apply & compare** also runs the comparison.

Note: when the scale is fitted, the comparison measures shape accuracy only, not whether
the scanner got the absolute size right.

The scan is aligned to the original automatically (rigid only; no scaling).
Both files are assumed to be in millimetres; you get a warning if their sizes differ a lot.

## Reading the results

| Column | Measures | Made worse by |
|---|---|---|
| **Scan → Original** | Distance from every scan point to the original surface | Untrimmed extra material on the scan |
| **Original → Scan** | Distance from every original point to the scan surface | Holes / missing areas in the scan |

If your scan is untrimmed, **Original → Scan** is the fairer accuracy number.
Signed values: **+** means the scan is larger than the original, **−** smaller.

## Development

```sh
.venv/bin/pytest --cov=mesh_compare
.venv/bin/mypy src tests
.venv/bin/ruff check src tests && .venv/bin/black --check src tests
```
