"""Entry point: ``python -m mesh_compare`` or the ``mesh-compare`` command."""

import sys

from PySide6.QtWidgets import QApplication

from mesh_compare.gui import MainWindow


def main() -> int:
    """Open the comparison window and run until it is closed."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
