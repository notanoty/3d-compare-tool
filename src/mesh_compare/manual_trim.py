"""The trimming window on its own: manual mode for the MCP server.

Opens the normal trimming window (optionally already holding proposed cuts), and
when it closes, trims and saves like the main window and writes a JSON report:

    python -m mesh_compare.manual_trim --model M.stl --result R.json \\
        [--other S.stl] [--apply-to-other] [--cuts '[{...}]'] [--up-axis +Z] \\
        [--output-dir D] [--output-name N] [--other-output-name N2] [--overwrite] [--no-images]

The report holds ``status`` ("saved", "skipped" or "error"), the cuts the user ended
with (as cut specs, see ``workflow``), the up axis, and the saved files.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog

from mesh_compare.comparison import load_mesh
from mesh_compare.trim_dialog import TrimDialog
from mesh_compare.workflow import VIEW_SIZE, TrimViews, cut_to_spec, parse_cut, trim_and_save


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--other", type=Path)
    parser.add_argument("--apply-to-other", action="store_true")
    parser.add_argument("--cuts", default="[]", help="JSON list of cut specs to start from")
    parser.add_argument("--up-axis", default="+Z")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-name")
    parser.add_argument("--other-output-name")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-images", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Show the window, then trim and save. Returns the report."""
    mesh = load_mesh(args.model)
    dialog = TrimDialog(
        mesh,
        args.model.name,
        other_name=args.other.name if args.other else None,
        view_size=VIEW_SIZE,
        up_axis=args.up_axis,
    )
    specs = json.loads(args.cuts)
    if specs:
        dialog.set_cuts([parse_cut(spec, dialog.cameras, mesh) for spec in specs])
    dialog.copy_box.setChecked(bool(args.apply_to_other and args.other))
    # Opened from the background by Claude: make sure it is seen.
    dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()

    if dialog.exec() != QDialog.DialogCode.Accepted:
        return {"status": "skipped"}

    up_axis = dialog.up_axis.currentText()
    cut_specs = [cut_to_spec(cut) for cut in dialog.cuts]
    report: dict[str, Any] = {
        "cuts": cut_specs,
        "up_axis": up_axis,
        "applied_to_other": dialog.copy_to_other,
    }
    views = TrimViews(mesh, up_axis, dialog.views, dialog.samples)  # type: ignore[arg-type]
    try:
        outcome = trim_and_save(
            views,
            dialog.cuts,
            args.model,
            args.other if dialog.copy_to_other else None,
            args.output_dir,
            args.output_name,
            args.other_output_name,
            args.overwrite,
            not args.no_images,
        )
    except (ValueError, OSError) as error:
        return {**report, "status": "error", "message": str(error)}
    return {
        **report,
        "status": "saved",
        "models": [str(path) for path in outcome.models],
        "images": [str(path) for path in outcome.images],
        "removed_pct": round(outcome.removed_fraction * 100, 2),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Mesh Compare — trim")
    try:
        report = run(args)
    except Exception as error:  # report every failure to the waiting MCP server
        traceback.print_exc()
        report = {"status": "error", "message": str(error)}
    partial = args.result.with_name(args.result.name + ".part")
    partial.write_text(json.dumps(report, indent=2), encoding="utf-8")
    partial.replace(args.result)
    return 0 if report["status"] != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
