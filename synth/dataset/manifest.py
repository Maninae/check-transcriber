"""The dataset manifest: everything needed to reproduce or audit a build."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MANIFEST_FILENAME = "manifest.json"


@dataclass
class DatasetManifest:
    """Build record written to `<output>/manifest.json`."""

    generator_version: str
    seed: int
    created_at: str
    command: str
    scene_counts: dict[str, int]
    check_counts: dict[str, int]
    template_ids_by_split: dict[str, list[str]]
    background_ids_by_split: dict[str, list[str]]
    background_root: str
    harmonized: bool
    harmonize_blend: float
    scene_config: dict
    split_fractions: dict[str, float]
    notes: list[str] = field(default_factory=list)


def write_manifest(manifest: DatasetManifest, output_directory: Path) -> Path:
    """Write the manifest as indented JSON; returns its path."""
    manifest_path = output_directory / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n")
    return manifest_path


def read_manifest(output_directory: Path) -> DatasetManifest:
    """Load a manifest written by `write_manifest`."""
    return DatasetManifest(**json.loads((output_directory / MANIFEST_FILENAME).read_text()))
