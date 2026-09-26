"""The field-reading area's one-heavy-MPS-job-at-a-time lock (a directory, created atomically by mkdir)."""

import contextlib
import logging
import time

from experiments.field_reading.field_localization.localization_config import MPS_LOCK_DIRECTORY

logger = logging.getLogger(__name__)

LOCK_POLL_SECONDS = 30


@contextlib.contextmanager
def hold_mps_lock(job_description: str):
    """Block until the lock directory can be created, hold it for the `with` body, then remove it."""
    while True:
        try:
            MPS_LOCK_DIRECTORY.mkdir()
            break
        except FileExistsError:
            logger.info("MPS lock held by another job; waiting %ds", LOCK_POLL_SECONDS)
            time.sleep(LOCK_POLL_SECONDS)
    (MPS_LOCK_DIRECTORY / "owner.txt").write_text(f"U2 field localization: {job_description}\n")
    logger.info("took MPS lock for %s", job_description)
    try:
        yield
    finally:
        (MPS_LOCK_DIRECTORY / "owner.txt").unlink(missing_ok=True)
        MPS_LOCK_DIRECTORY.rmdir()
        logger.info("released MPS lock")
