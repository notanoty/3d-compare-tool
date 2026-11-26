"""MCP server: lets Claude trim, scale and compare models and save every image.

Speaks the Model Context Protocol over stdio (JSON-RPC 2.0, one message per line)
with no extra dependencies. Start it with ``python -m mesh_compare.mcp_server`` (or
the ``mesh-compare-mcp`` command) from an MCP client such as the Claude desktop app.

Trimming has two modes:

* **Automatic**: ``preview_trim`` returns the four views with a pixel grid and a
  height ruler, so the model can be looked at and cuts placed and checked;
  ``trim_model`` then cuts and saves.
* **Manual**: ``open_manual_trim`` opens the normal trimming window on the user's
  screen (optionally holding proposed cuts); ``manual_trim_result`` waits for the
  user to finish and returns what was saved and the cuts they ended with.

Heavy modules (trimesh, Qt) load on the first tool call, so the server answers
the client's handshake straight away.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

SERVER_NAME = "mesh-compare"
SERVER_VERSION = "0.2.0"
PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
# Longest a single manual_trim_result call waits, so clients do not time out.
MAX_WAIT_SECONDS = 55.0

# Kept in step with trimming.VIEW_NAMES / UP_AXES (checked by the tests) so the
# tool list needs no heavy imports.
VIEW_NAMES = ("Front", "Right", "Back", "Left")
UP_AXES = ("+Z", "-Z", "+Y", "-Y", "+X", "-X")

Content = list[dict[str, Any]]
Arguments = dict[str, Any]


class RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


class ToolError(Exception):
    """A problem worth showing to the caller as a plain message (no traceback)."""


def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def png(data: bytes) -> dict[str, Any]:
    return {"type": "image", "data": base64.b64encode(data).decode("ascii"), "mimeType": "image/png"}


# ----------------------------------------------------------------------------- arguments


def _path(args: Arguments, key: str, required: bool = True) -> Path | None:
    value = args.get(key)
    if value in (None, ""):
        if required:
            raise ToolError(f"'{key}' is required")
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        raise ToolError(f"'{key}' must be an absolute path, got '{value}'")
    return path


def _file(args: Arguments, key: str, required: bool = True) -> Path | None:
    path = _path(args, key, required)
    if path is not None and not path.is_file():
        raise ToolError(f"File not found: {path}")
    return path


def _required_file(args: Arguments, key: str) -> Path:
    path = _file(args, key)
    assert path is not None
    return path


def _bool(args: Arguments, key: str, default: bool) -> bool:
    value = args.get(key, default)
    return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")


def _number(args: Arguments, key: str, default: float | None) -> float | None:
    value = args.get(key, default)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ToolError(f"'{key}' must be a number") from error


def _cuts(args: Arguments, required: bool) -> list[dict[str, Any]]:
    cuts = args.get("cuts") or []
    if isinstance(cuts, str):
        cuts = json.loads(cuts)
    if not isinstance(cuts, list) or not all(isinstance(cut, dict) for cut in cuts):
        raise ToolError("'cuts' must be a list of cut objects")
    if required and not cuts:
        raise ToolError("Give at least one cut")
    return cuts


def _str(args: Arguments, key: str, default: str | None = None) -> str | None:
    value = args.get(key, default)
    return None if value in (None, "") else str(value)


# ----------------------------------------------------------------------------- schemas

PATH = {"type": "string", "description": "Absolute path."}
UP_AXIS = {
    "type": "string",
    "enum": list(UP_AXES),
    "default": "+Z",
    "description": "Which model axis points up in the views; change it if the views are sideways.",
}
OVERWRITE = {
    "type": "boolean",
    "default": False,
    "description": "Replace files that already exist (otherwise the call fails and lists them).",
}
SAVE_IMAGES = {
    "type": "boolean",
    "default": True,
    "description": "Also save the picture(s) shown in the result next to the saved model.",
}
CUT = {
    "type": "object",
    "description": (
        "One cutting plane, drawn as a line on one view and extending straight back into it. "
        "Give either 'line' or 'height'."
    ),
    "properties": {
        "view": {"type": "string", "enum": list(VIEW_NAMES), "default": "Front"},
        "remove": {
            "type": "string",
            "enum": ["top", "bottom"],
            "description": "Which side of the line is cut away (relative to the view's up).",
        },
        "line": {
            "type": "array",
            "description": (
                "[[x1, y1], [x2, y2]] in pixels on that view (0-359, origin top-left), "
                "read off the preview grid."
            ),
            "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            "minItems": 2,
            "maxItems": 2,
        },
        "height": {
            "type": "number",
            "description": "Level cut at this fraction of the model height (0 = bottom, 1 = top), "
            "as on the ruler beside each view.",
        },
        "tilt_deg": {
            "type": "number",
            "description": "Tilt of a 'height' cut on its view, degrees counter-clockwise.",
        },
    },
    "required": ["remove"],
}
CUTS = {"type": "array", "items": CUT}


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


# ----------------------------------------------------------------------------- tools


def _workflow() -> Any:
    from mesh_compare import workflow

    workflow.ensure_qt()
    return workflow


def tool_list_models(args: Arguments) -> Content:
    folder = _path(args, "folder")
    assert folder is not None
    if not folder.is_dir():
        raise ToolError(f"Folder not found: {folder}")
    from mesh_compare.workflow import list_models

    models = list_models(folder, _bool(args, "recursive", False))
    if not models:
        return [text(f"No model files in {folder}")]
    lines = [f"{len(models)} model file(s) in {folder}:"]
    lines += [f"{path.relative_to(folder)}  ({path.stat().st_size / 1e6:.1f} MB)" for path in models]
    return [text("\n".join(lines))]


def tool_model_info(args: Arguments) -> Content:
    from mesh_compare.workflow import describe_model

    return [text(describe_model(_required_file(args, "path")))]


def tool_view_model(args: Arguments) -> Content:
    wf = _workflow()
    path = _required_file(args, "path")
    image = wf.model_views_image(path, args.get("up_axis") or "+Z")
    return [text(wf.describe_model(path)), png(wf.png_bytes(image))]


def tool_preview_trim(args: Arguments) -> Content:
    wf = _workflow()
    preview = wf.preview_trim(
        _required_file(args, "model"), _cuts(args, required=False), args.get("up_axis") or "+Z"
    )
    return [text(preview.text), png(wf.png_bytes(preview.image))]


def tool_trim_model(args: Arguments) -> Content:
    wf = _workflow()
    model_path = _required_file(args, "model")
    up_axis = args.get("up_axis") or "+Z"
    views = wf.TrimViews.render(wf.load_mesh(model_path), up_axis)
    cuts = views.cuts(_cuts(args, required=True))
    outcome = wf.trim_and_save(
        views,
        cuts,
        model_path,
        _file(args, "also_apply_to", required=False),
        _path(args, "output_dir", required=False),
        _str(args, "output_name"),
        _str(args, "other_output_name"),
        _bool(args, "overwrite", False),
        _bool(args, "save_images", True),
    )
    lines = [f"Removed about {outcome.removed_fraction * 100:.1f} % of {model_path.name}."]
    lines += [f"Saved model: {path}" for path in outcome.models]
    lines += [f"Saved image: {path}" for path in outcome.images]
    lines += ["The image(s) below show the trimmed result."]
    content = [text("\n".join(lines))]
    for path in outcome.models:
        content.append(png(wf.png_bytes(wf.model_views_image(path, up_axis, f"Trimmed: {path.name}"))))
    return content


@dataclass
class ManualJob:
    process: subprocess.Popen[bytes]
    result_path: Path
    log_path: Path
    up_axis: str


MANUAL_JOBS: dict[str, ManualJob] = {}


def tool_open_manual_trim(args: Arguments) -> Content:
    model = _required_file(args, "model")
    other = _file(args, "other", required=False)
    cuts = _cuts(args, required=False)
    up_axis = args.get("up_axis") or "+Z"
    folder = Path(tempfile.mkdtemp(prefix="mesh-compare-trim-"))
    job_id = uuid.uuid4().hex[:8]
    result_path = folder / "result.json"
    log_path = folder / "log.txt"
    command = [
        sys.executable, "-m", "mesh_compare.manual_trim",
        "--model", str(model),
        "--result", str(result_path),
        "--cuts", json.dumps(cuts),
        "--up-axis", up_axis,
    ]  # fmt: skip
    if other is not None:
        command += ["--other", str(other)]
        if _bool(args, "apply_to_other", False):
            command.append("--apply-to-other")
    output_dir = _path(args, "output_dir", required=False)
    if output_dir is not None:
        command += ["--output-dir", str(output_dir)]
    for key, flag in (("output_name", "--output-name"), ("other_output_name", "--other-output-name")):
        if (value := _str(args, key)) is not None:
            command += [flag, value]
    if _bool(args, "overwrite", False):
        command.append("--overwrite")
    if not _bool(args, "save_images", True):
        command.append("--no-images")

    # The window needs the real display, not the offscreen platform this server uses.
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    log = log_path.open("wb")
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env, start_new_session=True
    )
    log.close()
    MANUAL_JOBS[job_id] = ManualJob(process, result_path, log_path, up_axis)
    return [
        text(
            f"Opened the trimming window for {model.name} on the user's screen (job {job_id}). "
            "Ask the user to adjust the cuts and click 'Trim & save' (or 'Skip'), then call "
            f"manual_trim_result with job_id '{job_id}'."
        )
    ]


def tool_manual_trim_result(args: Arguments) -> Content:
    job_id = str(args.get("job_id") or "")
    job = MANUAL_JOBS.get(job_id)
    if job is None:
        raise ToolError(f"Unknown job '{job_id}'. Known: {', '.join(MANUAL_JOBS) or 'none'}")
    wait = min(max(_number(args, "wait_seconds", 30.0) or 0.0, 0.0), MAX_WAIT_SECONDS)
    deadline = time.monotonic() + wait
    while not job.result_path.exists():
        if job.process.poll() is not None:
            time.sleep(0.2)  # the report is written just before exit
            if job.result_path.exists():
                break
            log = job.log_path.read_text(errors="replace")[-2000:]
            raise ToolError(f"The trimming window closed without a report. Log:\n{log}")
        if time.monotonic() >= deadline:
            return [
                text(
                    "The trimming window is still open; the user has not finished. "
                    "Call manual_trim_result again to keep waiting."
                )
            ]
        time.sleep(0.25)

    report = json.loads(job.result_path.read_text(encoding="utf-8"))
    status = report.get("status")
    if status == "skipped":
        del MANUAL_JOBS[job_id]
        return [text("The user closed the trimming window with 'Skip'; nothing was saved.")]
    cuts = json.dumps(report.get("cuts", []))
    if status != "saved":
        del MANUAL_JOBS[job_id]
        return [
            text(
                f"Trimming failed: {report.get('message')}\n"
                f"The user's cuts (up axis {report.get('up_axis')}), reusable with trim_model: {cuts}"
            )
        ]
    del MANUAL_JOBS[job_id]
    lines = [
        f"The user trimmed and saved (removed about {report['removed_pct']} %).",
        *(f"Saved model: {path}" for path in report["models"]),
        *(f"Saved image: {path}" for path in report["images"]),
        f"Cuts used (up axis {report['up_axis']}), reusable with trim_model: {cuts}",
    ]
    wf = _workflow()
    content = [text("\n".join(lines))]
    for path in report["models"]:
        image = wf.model_views_image(Path(path), report["up_axis"], f"Trimmed: {Path(path).name}")
        content.append(png(wf.png_bytes(image)))
    return content


def _scale_content(outcome: Any, wf: Any) -> Content:
    lines = [
        f"Scale factor: {outcome.factor:.6g}",
        f"Saved model: {outcome.model}",
        f"Scaled size: {' × '.join(f'{v:.2f}' for v in outcome.scaled_size)} mm",
    ]
    if outcome.original_size is not None:
        lines.append(f"Original size: {' × '.join(f'{v:.2f}' for v in outcome.original_size)} mm")
    if outcome.alignment_rmse is not None:
        lines.append(f"Alignment fit after scaling (best 90 %): {outcome.alignment_rmse:.4f} mm")
    if outcome.image_path is not None:
        lines.append(f"Saved image: {outcome.image_path}")
    content = [text("\n".join(lines))]
    if outcome.image is not None:
        content.append(png(wf.png_bytes(outcome.image)))
    return content


def tool_auto_scale(args: Arguments) -> Content:
    wf = _workflow()
    outcome = wf.auto_scale(
        _required_file(args, "scan"),
        _required_file(args, "original"),
        _path(args, "output_dir", required=False),
        _str(args, "output_name"),
        _bool(args, "overwrite", False),
        _bool(args, "save_images", True),
        args.get("up_axis") or "+Z",
    )
    return _scale_content(outcome, wf)


def tool_scale_model(args: Arguments) -> Content:
    wf = _workflow()
    factor = _number(args, "factor", None)
    if factor is None:
        raise ToolError("'factor' is required")
    outcome = wf.scale_and_save(
        _required_file(args, "scan"),
        factor,
        _file(args, "original", required=False),
        _path(args, "output_dir", required=False),
        _str(args, "output_name"),
        _bool(args, "overwrite", False),
        _bool(args, "save_images", True),
        args.get("up_axis") or "+Z",
    )
    return _scale_content(outcome, wf)


def tool_compare_models(args: Arguments) -> Content:
    wf = _workflow()
    save = _bool(args, "save_files", True)
    outcome = wf.compare_and_save(
        _required_file(args, "original"),
        _required_file(args, "scan"),
        _number(args, "tolerance", 0.1),
        _path(args, "output_dir", required=False),
        _str(args, "results_name", "data.txt") if save else None,
        _str(args, "heatmap_name", "heatmap.png") if save else None,
        args.get("up_axis") or "+Z",
        _number(args, "missing_threshold", None),
        _bool(args, "align", True),
        _bool(args, "overwrite", False),
    )
    lines = [
        outcome.result.as_text(),
        "",
        f"Missing in scan (|deviation| > {outcome.missing_threshold:g} mm): "
        f"{outcome.missing_pct:.1f} % of the visible original surface",
    ]
    if outcome.results_path is not None:
        lines.append(f"Saved results: {outcome.results_path}")
    if outcome.heatmap_path is not None:
        lines.append(f"Saved heat map: {outcome.heatmap_path}")
    if outcome.detail_heatmap_path is not None:
        lines.append(f"Saved detail heat map: {outcome.detail_heatmap_path}")
    return [
        text("\n".join(lines)),
        png(wf.png_bytes(outcome.image)),
        png(wf.png_bytes(outcome.detail_image)),
    ]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    run: Callable[[Arguments], Content]


TOOLS = [
    Tool(
        "list_models",
        "List the 3D model files (STL, PLY, OBJ, OFF, glTF) in a folder.",
        _schema({"folder": PATH, "recursive": {"type": "boolean", "default": False}}, ["folder"]),
        tool_list_models,
    ),
    Tool(
        "model_info",
        "Size, triangle count, area and volume of a model file (mm).",
        _schema({"path": PATH}, ["path"]),
        tool_model_info,
    ),
    Tool(
        "view_model",
        "Render a model from front, right, back and left (e.g. to check orientation or a "
        "saved result).",
        _schema({"path": PATH, "up_axis": UP_AXIS}, ["path"]),
        tool_view_model,
    ),
    Tool(
        "preview_trim",
        "Show a model from four sides with a pixel grid and a height ruler, the given cuts drawn "
        "as coloured lines and everything they remove in red. Saves nothing. Call without cuts "
        "first to look at the model, then refine cuts until the red area is exactly the part to "
        "remove, then call trim_model with the same cuts.",
        _schema({"model": PATH, "cuts": CUTS, "up_axis": UP_AXIS}, ["model"]),
        tool_preview_trim,
    ),
    Tool(
        "trim_model",
        "Cut a model with the given cuts (same format as preview_trim) and save "
        "<name>_trimmed.stl plus <trimmed>_cuts.png. Optionally align another model and cut it "
        "at the same places. Returns renders of the trimmed result.",
        _schema(
            {
                "model": PATH,
                "cuts": CUTS,
                "up_axis": UP_AXIS,
                "also_apply_to": {**PATH, "description": "Another model to cut at the same places."},
                "output_dir": {**PATH, "description": "Folder to save into (default: next to "
                               "each source file)."},
                "output_name": {"type": "string", "description": "File name for the trimmed model."},
                "other_output_name": {"type": "string"},
                "overwrite": OVERWRITE,
                "save_images": SAVE_IMAGES,
            },
            ["model", "cuts"],
        ),
        tool_trim_model,
    ),
    Tool(
        "open_manual_trim",
        "Manual mode: open the trimming window on the user's screen so they can place or adjust "
        "the cuts themselves. Proposed cuts can be passed in as a starting point. Returns at once "
        "with a job id; then call manual_trim_result.",
        _schema(
            {
                "model": PATH,
                "cuts": CUTS,
                "up_axis": UP_AXIS,
                "other": {**PATH, "description": "Other model the user may apply the cuts to."},
                "apply_to_other": {"type": "boolean", "default": False,
                                   "description": "Pre-tick 'Also apply these cuts to …'."},
                "output_dir": PATH,
                "output_name": {"type": "string"},
                "other_output_name": {"type": "string"},
                "overwrite": OVERWRITE,
                "save_images": SAVE_IMAGES,
            },
            ["model"],
        ),
        tool_open_manual_trim,
    ),
    Tool(
        "manual_trim_result",
        "Wait (up to wait_seconds, max 55) for the user to finish in the trimming window opened "
        "by open_manual_trim. Returns the saved files, renders of the result and the cuts the "
        "user used, or says the window is still open.",
        _schema(
            {"job_id": {"type": "string"}, "wait_seconds": {"type": "number", "default": 30}},
            ["job_id"],
        ),
        tool_manual_trim_result,
    ),
    Tool(
        "auto_scale",
        "Estimate the uniform scale that makes the scan the original's size (like Auto-fit), "
        "save <scan>_scaled.stl and a check image of the scaled scan aligned on the original "
        "(grey overlap, orange = only original, blue = only scan).",
        _schema(
            {
                "scan": PATH,
                "original": PATH,
                "output_dir": PATH,
                "output_name": {"type": "string",
                                "description": "e.g. scan-trimmed_auto-scaled.stl"},
                "up_axis": UP_AXIS,
                "overwrite": OVERWRITE,
                "save_images": SAVE_IMAGES,
            },
            ["scan", "original"],
        ),
        tool_auto_scale,
    ),
    Tool(
        "scale_model",
        "Scale a model by an exact factor (e.g. 1000 for m → mm) and save it; with an original, "
        "also make the same check image as auto_scale.",
        _schema(
            {
                "scan": PATH,
                "factor": {"type": "number"},
                "original": PATH,
                "output_dir": PATH,
                "output_name": {"type": "string"},
                "up_axis": UP_AXIS,
                "overwrite": OVERWRITE,
                "save_images": SAVE_IMAGES,
            },
            ["scan", "factor"],
        ),
        tool_scale_model,
    ),
    Tool(
        "compare_models",
        "Align the scan to the original and measure deviations, like the Compare button. Saves "
        "the results table (data.txt), the heat map image (heatmap.png) and a detail heat map "
        "(heatmap_detail.png: green at the best-matching point, red at the largest non-missing "
        "deviation, so differences inside the tolerance show) into output_dir "
        "(default: the scan's folder) and returns all three.",
        _schema(
            {
                "original": PATH,
                "scan": PATH,
                "tolerance": {"type": "number", "default": 0.1, "description": "± mm"},
                "output_dir": PATH,
                "results_name": {"type": "string", "default": "data.txt"},
                "heatmap_name": {"type": "string", "default": "heatmap.png"},
                "up_axis": UP_AXIS,
                "missing_threshold": {"type": "number",
                                      "description": "mm; default max(0.5, tolerance + 0.01)"},
                "align": {"type": "boolean", "default": True},
                "save_files": {"type": "boolean", "default": True},
                "overwrite": OVERWRITE,
            },
            ["original", "scan"],
        ),
        tool_compare_models,
    ),
]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


# ----------------------------------------------------------------------------- protocol


class Server:
    """Handles one JSON-RPC message at a time and writes replies to ``out``."""

    def __init__(self, out: IO[str]) -> None:
        self.out = out

    def send(self, message: dict[str, Any]) -> None:
        self.out.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.out.flush()

    def handle(self, message: Any) -> None:
        if not isinstance(message, dict) or "method" not in message:
            return  # responses to requests we never send, or junk
        request_id = message.get("id")
        if request_id is None:
            return  # notifications (initialized, cancelled, …) need no reply
        try:
            result = self.dispatch(str(message["method"]), message.get("params") or {})
        except RpcError as error:
            self.send(
                {"jsonrpc": "2.0", "id": request_id, "error": {"code": error.code, "message": str(error)}}
            )
        else:
            self.send({"jsonrpc": "2.0", "id": request_id, "result": result})

    def dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            requested = params.get("protocolVersion")
            version = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
            return {
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "instructions": __doc__,
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {
                "tools": [
                    {"name": t.name, "description": t.description, "inputSchema": t.schema}
                    for t in TOOLS
                ]
            }
        if method == "tools/call":
            return self.call_tool(str(params.get("name")), params.get("arguments") or {})
        if method in ("resources/list", "prompts/list"):
            return {method.split("/")[0]: []}
        raise RpcError(-32601, f"Method not found: {method}")

    def call_tool(self, name: str, arguments: Arguments) -> dict[str, Any]:
        tool = TOOLS_BY_NAME.get(name)
        if tool is None:
            raise RpcError(-32602, f"Unknown tool: {name}")
        try:
            content = tool.run(arguments)
        except ToolError as error:
            return {"content": [text(f"Error: {error}")], "isError": True}
        except (ValueError, OSError) as error:  # bad cuts, missing/existing files, …
            return {"content": [text(f"Error: {error}")], "isError": True}
        except Exception as error:  # keep serving; show the trace to whoever reads it
            traceback.print_exc()
            return {
                "content": [text(f"Error ({type(error).__name__}): {error}")],
                "isError": True,
            }
        return {"content": content}


def main() -> int:
    """Serve on stdin/stdout until stdin closes."""
    out = open(sys.stdout.fileno(), "w", encoding="utf-8", closefd=False)  # noqa: SIM115
    # Libraries that print would corrupt the protocol stream; send them to stderr.
    sys.stdout = sys.stderr
    server = Server(out)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            server.send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
            continue
        for item in message if isinstance(message, list) else [message]:
            server.handle(item)
    return 0


if __name__ == "__main__":
    sys.exit(main())
