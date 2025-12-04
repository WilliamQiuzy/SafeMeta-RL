"""Episode-level logger for immediate recording after each episode."""

import csv
from pathlib import Path
from typing import Dict, Any


class EpisodeLogger:
    """Log metrics immediately after each episode completes."""

    def __init__(self, log_dir: Path):
        self.log_path = log_dir / "episodes.csv"
        self.file_handle = None
        self.csv_writer = None
        self.episode_count = 0

    def log_episode(self, metrics: Dict[str, Any]) -> None:
        """Log a single episode's metrics."""
        metrics["episode_num"] = self.episode_count
        self.episode_count += 1

        if self.file_handle is None:
            # Initialize CSV file
            self.file_handle = open(self.log_path, 'w', newline='')
            self.csv_writer = csv.DictWriter(
                self.file_handle,
                fieldnames=list(metrics.keys())
            )
            self.csv_writer.writeheader()

        self.csv_writer.writerow(metrics)
        self.file_handle.flush()  # Immediate write

    def close(self) -> None:
        if self.file_handle:
            self.file_handle.close()