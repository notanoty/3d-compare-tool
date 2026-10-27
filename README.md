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
