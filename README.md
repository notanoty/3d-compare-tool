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

**Save new models to** (main window): pick a folder with **Choose folder…** and every
trimmed / scaled model and the default heat map image go there. **Next to model** goes back
to saving beside the source file. The choice is remembered between runs.

1. Drop the **original model** into the left box and the **new scan** into the right box
   (or click a box to browse). Supported: STL, PLY, OBJ, OFF, glTF.
2. Set the tolerance (default ±0.1 mm) and click **Compare**.
3. **Copy results** puts a tab-separated table on the clipboard for Excel / Google Sheets.
4. **Heat map…** (ready a moment after the numbers) shows where the scan deviates — see below.

### Trimming

When a model is added, a trimming window opens (turn off with *Open trimming when a model
is added*, or use the **Trim…** buttons later). It shows the model from the front, right,
back and left.

- Drag on any view to draw the selected cut line; drag a line's round ends to tilt or move it.
- **Add cut** adds another line (e.g. one for the base, one for the canal tip); each cut has
  its own colour and its own **Remove top / Remove bottom** setting. Select a cut in the
  list (or grab its handle) to change it; **Delete cut** removes it.
- Red shows everything that will be removed by all cuts together, on every view.
- **Up axis** changes which way is "up" if the model is lying on its side.
- **Also apply these cuts to …** aligns the other model and cuts it at the same places.
- **Trim & save** writes `<name>_trimmed.stl` into the *Save new models to* folder (the
  original is never changed). Comparisons then use the trimmed files.

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

### Heat map

After each comparison a deviation heat map is built: the original model seen from the front,
right, back and left, coloured by how far the scan is from each point.

- **Green**: within the tolerance.
- **Yellow → red**: scan larger (outside the original); **cyan → blue**: scan smaller.
- **Magenta**: missing in the scan — the scan is further away than *Missing beyond*
  (default 0.5 mm). The caption gives the missing share of the
  visible surface.

Change the **Up axis** if the views are sideways, adjust **Missing beyond**, then
**Save image…** to download it as PNG or JPG (with legend and file names).

## Let Claude do it (MCP server)

`mesh_compare.mcp_server` is an [MCP](https://modelcontextprotocol.io) server, so Claude can
trim, scale and compare models and save every image itself. It needs nothing beyond the
normal install.

Add it to the Claude desktop app (*Settings → Developer → Edit Config*, file
`claude_desktop_config.json`) and restart the app:

```json
{
  "mcpServers": {
    "mesh-compare": {
      "command": "/Users/notanoty/projects/dev/3d-compare-tool/.venv/bin/python",
      "args": ["-m", "mesh_compare.mcp_server"]
    }
  }
}
```

(After `pip install -e .` the `mesh-compare-mcp` command does the same.)

| Tool | Does |
|---|---|
| `list_models`, `model_info`, `view_model` | Find model files, sizes, four-side render |
| `preview_trim` | Four views with pixel grid, height ruler, cut lines and the removed part in red. Saves nothing |
| `trim_model` | Cut and save `<name>_trimmed.stl` + `<trimmed>_cuts.png`; optionally the other model at the same places |
| `open_manual_trim` / `manual_trim_result` | **Manual mode**: opens the trimming window on your screen (with Claude's proposed cuts), you adjust and click *Trim & save*, Claude gets the files and the cuts you used |
| `auto_scale` | Auto-fit scale, save `<scan>_scaled.stl` + `_scale-check.png` (grey overlap, orange only original, blue only scan) |
| `scale_model` | Scale by an exact factor (e.g. × 1000) |
| `compare_models` | Compare, save `data.txt` + `heatmap.png` into the chosen folder |

Cuts are JSON, either a level line at a fraction of the model height or a pixel line on a
view (the same 360 px views as the trimming window, so cuts move between the two):

```json
[{"view": "Front", "remove": "bottom", "height": 0.15, "tilt_deg": 5},
 {"view": "Right", "remove": "top", "line": [[40, 70], [320, 95]]}]
```

Nothing is overwritten unless the call says `overwrite: true`.

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

## Detail heat map

Besides the standard heat map (green inside ±tolerance, yellow/red and cyan/blue
up to the missing threshold), Compare also saves `heatmap_detail.png`. The detail
map has no flat green band. Colours run dark green → green → lime → yellow →
orange → red over the deviations that actually occur:

- the best-matching point of the original gets the first colour. That colour is
  dark green when the best point is close to the original and moves up the ramp
  (towards yellow) when even the best point is far off;
- red is the 99th percentile of the deviations that are not missing, so a few stray
  points do not flatten everything else;
- missing areas (beyond the missing threshold) stay magenta.

The colour shows the size of the deviation, not its sign; use the standard map for
"scan larger / smaller". In the heat map window, switch with **Colours: Tolerance /
Detail (relative)**; Save image suggests `<name>_detail.png` for the detail view.
