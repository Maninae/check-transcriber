"""The dataset manifest: everything needed to reproduce or audit a build.

- `split_pools[split]` holds that split's template, background, handwriting-font and
  signature-font ids (contract C4); the same dict is written up front as `build_plan.json`
  so a resumed build can prove it uses the same pools.
- `git_commit` is the generator's commit, suffixed `-dirty` when the working tree had changes.
"""

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

MANIFEST_FILENAME = "manifest.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GIT_TIMEOUT_SECONDS = 10


@dataclass
class DatasetManifest:
    """Build record written to `<output>/manifest.json`."""

    generator_version: str
    git_commit: str
    seed: int
    created_at: str
    command: str
    workers: int
    template_count: int
    scene_counts: dict[str, int]
    check_counts: dict[str, int]
    failed_scene_ids: list[str]
    split_pools: dict[str, dict[str, list[str]]]
    split_fractions: dict[str, float]
    background_root: str
    harmonized: bool
    harmonize_blend: float
    scene_config: dict
    ocr_splits: list[str]
    ocr_row_counts: dict[str, dict[str, int]]
    timings_seconds: dict[str, float]
    notes: list[str] = field(default_factory=list)


def current_git_commit(repository_root: Path = REPOSITORY_ROOT) -> str:
    """HEAD of the generator's repository (`-dirty` if uncommitted changes), or `unknown` outside git."""
    try:
        commit = subprocess.run(["git", "-C", str(repository_root), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True, timeout=GIT_TIMEOUT_SECONDS).stdout.strip()
        status = subprocess.run(["git", "-C", str(repository_root), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, check=True, timeout=GIT_TIMEOUT_SECONDS).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{commit}-dirty" if status else commit


def write_manifest(manifest: DatasetManifest, output_directory: Path) -> Path:
    """Write the manifest as indented JSON; returns its path."""
    manifest_path = output_directory / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n")
    return manifest_path


def read_manifest(output_directory: Path) -> DatasetManifest:
    """Load a manifest written by `write_manifest`."""
    return DatasetManifest(**json.loads((output_directory / MANIFEST_FILENAME).read_text()))
