"""Lightweight experiment logger."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Any


class CSVLogger:
    """Appends metrics to a CSV file for later analysis."""

    def __init__(self, log_dir: Path, filename: str = "metrics.csv", resume: bool = False):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.log_dir / filename
        self._writer = None
        self._file = None
        self._resume = resume
        self._fieldnames = None

    def set_resume(self, resume: bool) -> None:
        """Enable resume mode to append to existing CSV instead of overwriting."""
        self._resume = resume

    def log(self, metrics: Dict[str, Any]) -> None:
        if self._writer is None:
            fieldnames = list(metrics.keys())
            # Check if file exists and we're in resume mode
            if self._resume and self.file_path.exists():
                # Read existing fieldnames from the file
                with self.file_path.open("r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    existing_fieldnames = next(reader, None)
                    if existing_fieldnames:
                        fieldnames = existing_fieldnames
                # Open in append mode
                self._file = self.file_path.open("a", newline="", encoding="utf-8")
                self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
                # Don't write header again
            else:
                # Normal mode: overwrite file
                self._file = self.file_path.open("w", newline="", encoding="utf-8")
                self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
                self._writer.writeheader()
            self._fieldnames = fieldnames
        self._writer.writerow(metrics)
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None
