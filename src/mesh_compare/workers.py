"""Background work for the Qt windows."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QThread, Signal


class TaskWorker(QThread):
    """Runs a slow task off the UI thread so the window stays responsive."""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, task: Callable[[], object]) -> None:
        super().__init__()
        self.task = task

    def run(self) -> None:
        try:
            result = self.task()
        except Exception as error:  # report any failure in the window instead of crashing
            self.failed.emit(str(error))
        else:
            self.succeeded.emit(result)
