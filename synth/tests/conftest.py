"""Shared fixtures: one small dataset build per test session, reused by every stage-3 test module."""

import pytest

from synth.backgrounds.procedural import write_procedural_backgrounds
from synth.dataset.build_dataset import build_dataset, parse_arguments

BUILD_SCENES = 14   # 10 train, 2 val, 2 eval: every split non-empty, OCR splits included


@pytest.fixture(scope="session")
def built_dataset(tmp_path_factory):
    """Build a 14-scene dataset once; returns (root, output dir, manifest, parsed arguments)."""
    root = tmp_path_factory.mktemp("build")
    write_procedural_backgrounds(root / "backgrounds" / "fabric", 6, seed=3)
    arguments = parse_arguments(["--output", str(root / "out"), "--scenes", str(BUILD_SCENES), "--seed", "3",
                                 "--workers", "2", "--backgrounds", str(root / "backgrounds")])
    manifest = build_dataset(arguments)
    return root, root / "out", manifest, arguments
