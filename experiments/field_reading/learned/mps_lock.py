"""The area-wide MPS lock (PLAN.md): one heavy MPS job at a time for field reading.

The lock is an empty directory created with `mkdir` (atomic). Holders must `rmdir` it on exit,
including on failure, which the context manager guarantees for Python jobs.
"""

import contextlib
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator

from experiments.field_reading.config import FIELD_READING_OUTPUT_ROOT

logger = logging.getLogger(__name__)

MPS_LOCK_PATH = FIELD_READING_OUTPUT_ROOT / "logs" / "MPS_LOCK"
LOCK_RETRY_SECONDS = 120


def log_memory_pressure() -> None:
    """Log the free/inactive page counts from `vm_stat` so a run's memory context is on record."""
    output = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False).stdout
    summary = [line.strip() for line in output.splitlines() if line.startswith(("Pages free", "Pages inactive"))]
    logger.info("vm_stat: %s", "; ".join(summary))


@contextlib.contextmanager
def hold_mps_lock(job_name: str) -> Iterator[None]:
    """Block until the lock directory can be created, hold it for the body, always remove it."""
    log_memory_pressure()
    while True:
        try:
            os.mkdir(MPS_LOCK_PATH)
            break
        except FileExistsError:
            logger.info("%s: MPS lock held by another job, retrying in %ds", job_name, LOCK_RETRY_SECONDS)
            time.sleep(LOCK_RETRY_SECONDS)
    logger.info("%s: acquired MPS lock", job_name)
    # `kill` (SIGTERM) must still run the finally below, or the lock outlives the job.
    signal.signal(signal.SIGTERM, lambda signal_number, frame: sys.exit(128 + signal_number))
    try:
        yield
    finally:
        os.rmdir(MPS_LOCK_PATH)
        logger.info("%s: released MPS lock", job_name)
