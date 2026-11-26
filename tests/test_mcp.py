"""Tests for the scriptable workflow, the stand-alone trimming window and the MCP server."""

import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import trimesh

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from mesh_compare import manual_trim, mcp_server, trimming, workflow  # noqa: E402
from mesh_compare.trim_dialog import TrimDialog  # noqa: E402


@pytest.fixture(scope="module")
def app() -> QApplication:
    existing = QApplication.instance()
    return existing if isinstance(existing, QApplication) else QApplication([])


@pytest.fixture
def model(tmp_path: Path, lumpy_mesh: trimesh.Trimesh) -> Path:
    path = tmp_path / "model.stl"
    lumpy_mesh.export(path)
    return path


@pytest.fixture
def small_scan(tmp_path: Path, lumpy_mesh: trimesh.Trimesh) -> Path:
    """The same shape at 1/1000 size, moved and turned (a scan saved in metres)."""
    scan = lumpy_mesh.copy()
    scan.apply_transform(trimesh.transformations.euler_matrix(0.2, -0.1, 0.3))
    scan.apply_translation((5.0, -3.0, 2.0))
    scan.apply_scale(0.001)
    path = tmp_path / "scan.stl"
    scan.export(path)
    return path


BOTTOM_CUT = {"view": "Front", "remove": "bottom", "height": 0.2}


def test_server_constants_match_trimming() -> None:
    assert mcp_server.VIEW_NAMES == trimming.VIEW_NAMES
    assert mcp_server.UP_AXES == tuple(trimming.UP_AXES)


def test_height_and_line_cuts_agree(app: QApplication, lumpy_mesh: trimesh.Trimesh) -> None:
    views = workflow.TrimViews.render(lumpy_mesh)
    (by_height,) = views.cuts([BOTTOM_CUT])
    as_line = workflow.cut_to_spec(by_height)
    (by_line,) = views.cuts([as_line])
    removed = [views.samples.removed_fraction(views.planes([c])) for c in (by_height, by_line)]
    assert removed[0] == pytest.approx(removed[1], abs=1e-3)
    assert 0.05 < removed[0] < 0.5


def test_tilt_turns_the_line_counter_clockwise(lumpy_mesh: trimesh.Trimesh) -> None:
    camera = trimming.side_cameras(lumpy_mesh)[0]
    (x1, y1), (x2, y2) = workflow.level_line(camera, lumpy_mesh, 0.5, tilt_deg=20)
    assert x2 > x1 and y2 < y1  # right end higher on screen


@pytest.mark.parametrize(
    "spec",
    [
        {"remove": "bottom"},
        {"height": 0.2},
        {"view": "Top", "remove": "top", "height": 0.5},
        {"remove": "top", "line": [[1, 2]]},
    ],
)
def test_bad_cuts_are_refused(lumpy_mesh: trimesh.Trimesh, spec: dict[str, object]) -> None:
    cameras = trimming.side_cameras(lumpy_mesh)
    with pytest.raises(ValueError):
        workflow.parse_cut(spec, cameras, lumpy_mesh)


def test_preview_saves_nothing(app: QApplication, model: Path) -> None:
    preview = workflow.preview_trim(model, [BOTTOM_CUT])
    assert "Removes about" in preview.text
    assert preview.image.width() > 2 * workflow.VIEW_SIZE
    assert workflow.png_bytes(preview.image)[:8] == b"\x89PNG\r\n\x1a\n"
    assert sorted(p.name for p in model.parent.iterdir()) == ["model.stl"]


def test_trim_saves_model_and_image_and_protects_files(app: QApplication, model: Path) -> None:
    views = workflow.TrimViews.render(trimesh.load(model))
    cuts = views.cuts([BOTTOM_CUT])
    outcome = workflow.trim_and_save(views, cuts, model)
    assert [p.name for p in outcome.models] == ["model_trimmed.stl"]
    assert [p.name for p in outcome.images] == ["model_trimmed_cuts.png"]
    trimmed = trimesh.load(outcome.models[0])
    assert trimmed.area < views.mesh.area
    with pytest.raises(FileExistsError):
        workflow.trim_and_save(views, cuts, model)
    workflow.trim_and_save(views, cuts, model, overwrite=True)


def test_trim_applies_cuts_to_other_model(app: QApplication, model: Path, tmp_path: Path) -> None:
    other = tmp_path / "other.stl"
    moved = trimesh.load(model)
    moved.apply_transform(trimesh.transformations.euler_matrix(0.3, 0.2, -0.4))
    moved.export(other)
    views = workflow.TrimViews.render(trimesh.load(model))
    out = tmp_path / "out"
    outcome = workflow.trim_and_save(
        views, views.cuts([BOTTOM_CUT]), model, other, out, "a.stl", "b"
    )
    assert [p.name for p in outcome.models] == ["a.stl", "b.stl"]
    areas = [trimesh.load(p).area for p in outcome.models]
    assert areas[0] == pytest.approx(areas[1], rel=0.03)


def test_auto_scale_finds_unit_mix_up(app: QApplication, model: Path, small_scan: Path) -> None:
    outcome = workflow.auto_scale(small_scan, model, output_name="scan_auto-scaled")
    assert outcome.factor == pytest.approx(1000, rel=0.01)
    assert outcome.model.name == "scan_auto-scaled.stl"
    assert outcome.image_path is not None and outcome.image_path.exists()
    assert outcome.alignment_rmse is not None and outcome.alignment_rmse < 0.1


def test_compare_saves_results_and_heatmap(app: QApplication, model: Path, tmp_path: Path) -> None:
    out = tmp_path / "results"
    outcome = workflow.compare_and_save(model, model, tolerance=0.2, output_dir=out)
    assert outcome.result.scan_to_original.rms < 0.05
    assert (out / "data.txt").read_text().startswith("Metric")
    assert (out / "heatmap.png").exists()
    assert (out / "heatmap_detail.png").exists()
    with pytest.raises(FileExistsError):
        workflow.compare_and_save(model, model, output_dir=out)


def test_manual_trim_saves_what_the_user_set(
    app: QApplication, model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_exec(dialog: TrimDialog) -> QDialog.DialogCode:
        seen["cuts"] = len(dialog.cuts)
        dialog.set_remove("bottom")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(TrimDialog, "exec", fake_exec)
    monkeypatch.setattr(TrimDialog, "show", lambda dialog: None)
    args = argparse.Namespace(
        model=model, result=tmp_path / "r.json", other=None, apply_to_other=False,
        cuts=json.dumps([BOTTOM_CUT, {"remove": "top", "height": 0.9}]), up_axis="+Z",
        output_dir=None, output_name=None, other_output_name=None, overwrite=False,
        no_images=False,
    )  # fmt: skip
    report = manual_trim.run(args)
    assert seen["cuts"] == 2
    assert report["status"] == "saved"
    assert len(report["cuts"]) == 2 and report["cuts"][0]["view"] == "Front"
    assert Path(report["models"][0]).exists()


def test_manual_trim_skip(app: QApplication, model: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TrimDialog, "exec", lambda dialog: QDialog.DialogCode.Rejected)
    monkeypatch.setattr(TrimDialog, "show", lambda dialog: None)
    args = manual_trim._parser().parse_args(["--model", str(model), "--result", "unused.json"])
    assert manual_trim.run(args) == {"status": "skipped"}


def _rpc(lines: list[dict[str, object]]) -> list[dict[str, object]]:
    src = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": str(src)}
    stdin = "".join(json.dumps(line) + "\n" for line in lines)
    done = subprocess.run(
        [sys.executable, "-m", "mesh_compare.mcp_server"],
        input=stdin, capture_output=True, text=True, env=env, timeout=300, check=True,
    )  # fmt: skip
    return [json.loads(line) for line in done.stdout.splitlines()]


def test_server_end_to_end(model: Path) -> None:
    replies = _rpc(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                        "clientInfo": {"name": "test", "version": "0"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "preview_trim",
                        "arguments": {"model": str(model), "cuts": [BOTTOM_CUT]}}},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
             "params": {"name": "trim_model", "arguments": {"model": str(model), "cuts": []}}},
        ]
    )  # fmt: skip
    by_id = {reply["id"]: reply for reply in replies}
    assert by_id[1]["result"]["protocolVersion"] == "2025-06-18"  # type: ignore[index]
    names = [tool["name"] for tool in by_id[2]["result"]["tools"]]  # type: ignore[index]
    assert "preview_trim" in names and "compare_models" in names
    content = by_id[3]["result"]["content"]  # type: ignore[index]
    assert content[0]["type"] == "text" and content[1]["type"] == "image"
    assert base64.b64decode(content[1]["data"])[:4] == b"\x89PNG"
    assert by_id[4]["result"]["isError"] is True  # type: ignore[index]
