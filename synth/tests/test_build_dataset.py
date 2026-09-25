"""Stage 3 invariants: split rules hold, the manifest round-trips, a 20-scene build completes."""

import json

import numpy as np
import pytest

from synth.backgrounds.procedural import write_procedural_backgrounds
from synth.dataset.build_dataset import build_dataset, parse_arguments
from synth.dataset.manifest import read_manifest
from synth.dataset.splits import DEFAULT_SPLIT_FRACTIONS, SPLIT_NAMES, assign_ids_to_splits


@pytest.fixture(scope="module")
def built_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("build")
    write_procedural_backgrounds(root / "backgrounds" / "fabric", 6, seed=3)
    arguments = parse_arguments(["--output", str(root / "out"), "--scenes", "20", "--seed", "3", "--workers", "4",
                                 "--backgrounds", str(root / "backgrounds")])
    manifest = build_dataset(arguments)
    return root / "out", manifest


def test_twenty_scene_build_completes(built_dataset):
    output_directory, manifest = built_dataset
    images = sorted(output_directory.glob("*/images/*.jpg"))
    assert len(images) == 20 == sum(manifest.scene_counts.values())
    for image_path in images:
        split_directory = image_path.parent.parent
        assert (split_directory / "annotations" / f"{image_path.stem}.json").exists()
        assert (split_directory / "labels" / f"{image_path.stem}.txt").exists()
        assert (split_directory / "labels_obb" / f"{image_path.stem}.txt").exists()
    for split_name in SPLIT_NAMES:
        coco = json.loads((output_directory / split_name / "annotations_coco.json").read_text())
        assert len(coco["images"]) == manifest.scene_counts[split_name]
    assert (output_directory / "data.yaml").exists()


def test_no_template_or_background_shared_across_splits(built_dataset):
    output_directory, manifest = built_dataset
    for assignment in (manifest.template_ids_by_split, manifest.background_ids_by_split):
        seen = [set(assignment[name]) for name in SPLIT_NAMES]
        assert not (seen[0] & seen[1]) and not (seen[0] & seen[2]) and not (seen[1] & seen[2])
    for split_name in SPLIT_NAMES:
        for label_path in (output_directory / split_name / "annotations").glob("*.json"):
            scene = json.loads(label_path.read_text())
            assert scene["background_id"] in manifest.background_ids_by_split[split_name]
            assert {check["template_id"] for check in scene["checks"]} <= set(manifest.template_ids_by_split[split_name])


def test_manifest_round_trips(built_dataset):
    output_directory, manifest = built_dataset
    assert read_manifest(output_directory) == manifest


def test_split_assignment_is_a_partition():
    ids = [f"id_{i}" for i in range(24)]
    splits = assign_ids_to_splits(ids, DEFAULT_SPLIT_FRACTIONS, seed=1, salt=2)
    assert sorted(sum(splits.values(), [])) == sorted(ids)
    assert all(splits[name] for name in SPLIT_NAMES)
    assert splits == assign_ids_to_splits(ids, DEFAULT_SPLIT_FRACTIONS, seed=1, salt=2)
    assert np.isclose(len(splits["train"]) / len(ids), 0.8, atol=0.1)
