"""Progress, ETA and failure logging for a long parallel build stage.

The ETA uses only the work actually done in this run (skipped, already-built items are excluded
from the rate), so a resumed build does not report an absurdly optimistic ETA.
"""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PROGRESS_REPORTS_PER_STAGE = 20
FAILURE_LOG_FILENAME = "failures.jsonl"


class StageProgress:
    """Counts built / skipped / failed items of one stage and prints a line every ~5%."""

    def __init__(self, stage_name: str, total: int, output_directory: Path):
        self.stage_name = stage_name
        self.total = total
        self.failure_log_path = output_directory / FAILURE_LOG_FILENAME
        self.counts = {"built": 0, "skipped": 0, "failed": 0}
        self.failed_ids: list[str] = []
        self.start_time = time.time()
        self.report_every = max(1, total // PROGRESS_REPORTS_PER_STAGE)

    @property
    def completed(self) -> int:
        """Items finished in any state."""
        return sum(self.counts.values())

    def record(self, item_id: str, status: str, error_record: dict | None = None) -> None:
        """Count one finished item; failures are appended to `failures.jsonl` with their id."""
        self.counts[status] += 1
        if status == "failed":
            self.failed_ids.append(item_id)
            with open(self.failure_log_path, "a") as failure_log:
                failure_log.write(json.dumps({"stage": self.stage_name, "id": item_id, **(error_record or {})}) + "\n")
            logger.warning("%s failed for %s: %s", self.stage_name, item_id, (error_record or {}).get("error"))
        if self.completed % self.report_every == 0 or self.completed == self.total:
            self.print_line()

    def print_line(self) -> None:
        """One progress line with rate and ETA."""
        elapsed = time.time() - self.start_time
        worked = self.counts["built"] + self.counts["failed"]
        rate = worked / elapsed if elapsed > 0 else 0.0
        remaining = self.total - self.completed
        eta = f"{remaining / rate:.0f}s" if rate > 0 else "?"
        print(f"[{self.stage_name} {self.completed}/{self.total}] built {self.counts['built']}, "
              f"skipped {self.counts['skipped']}, failed {self.counts['failed']}, {rate:.2f}/s, eta {eta}", flush=True)

    def elapsed_seconds(self) -> float:
        """Wall time since the stage started."""
        return round(time.time() - self.start_time, 1)
