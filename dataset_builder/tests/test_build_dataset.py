"""Stage 3 invariants: split pools are disjoint and respected, the manifest round-trips, builds resume."""

import json
from dataclasses import replace

import numpy as np
import pytest

from dataset_builder.build_dataset import build_dataset, parse_arguments, scene_tasks_for_plan
from dataset_builder.build_plan import BuildPlan
from dataset_builder.manifest import read_manifest
from dataset_builder.scene_task_runner import run_scene_task
from dataset_builder.tests.conftest import BUILD_SCENES
from scene_composer.on_demand import SyntheticSceneStream
from synthetic_checks.check_templates import build_template_catalog, template_family_by_id
from synthetic_checks.splits import DEFAULT_SPLIT_FRACTIONS, SPLIT_NAMES, assign_ids_by_hash_rank, plan_split_pools, plan_template_pools


def test_build_writes_every_per_scene_file(built_dataset):
    _, output_directory, manifest, _ = built_dataset
    images = sorted(output_directory.glob("*/images/*.jpg"))
    assert len(images) == BUILD_SCENES == sum(manifest.scene_counts.values())
    assert manifest.failed_scene_ids == []
    for image_path in images:
        split_directory = image_path.parent.parent
        for subdirectory, suffix in (("annotations", "json"), ("labels", "txt"), ("labels_obb", "txt")):
            assert (split_directory / subdirectory / f"{image_path.stem}.{suffix}").exists()
    assert not list(output_directory.glob("*/annotations/*.partial"))


def test_no_id_of_any_kind_crosses_splits(built_dataset):
    _, _, manifest, _ = built_dataset
    assert {"payee_names", "bank_names"} <= set(manifest.split_pools["train"])
    for kind in manifest.split_pools["train"]:
        pools = [set(manifest.split_pools[name][kind]) for name in SPLIT_NAMES]
        assert all(pools), kind
        assert not (pools[0] & pools[1]) and not (pools[0] & pools[2]) and not (pools[1] & pools[2]), kind


def test_scenes_only_use_their_own_split_pools(built_dataset):
    _, output_directory, manifest, _ = built_dataset
    used_handwriting_fonts = set()
    for split_name in SPLIT_NAMES:
        pools = manifest.split_pools[split_name]
        for label_path in (output_directory / split_name / "annotations").glob("*.json"):
            scene = json.loads(label_path.read_text())
            assert scene["background_id"] in pools["background_ids"]
            for check in scene["checks"]:
                assert check["template_id"] in pools["template_ids"]
                assert check["canonical"]["handwriting_font_id"] in pools["handwriting_font_ids"]
                assert check["canonical"]["signature_font_id"] in pools["signature_font_ids"]
                assert check["canonical"]["payee_canonical"] in pools["payee_names"]
                bank_texts = [field["text"] for field in check["fields"] if field["field_name"] == "bank_name"]
                assert bank_texts and all(text in pools["bank_names"] for text in bank_texts)
                used_handwriting_fonts.add(check["canonical"]["handwriting_font_id"])
    assert len(used_handwriting_fonts) > 1


def test_manifest_round_trips_and_records_the_build(built_dataset):
    _, output_directory, manifest, _ = built_dataset
    assert read_manifest(output_directory) == manifest
    assert manifest.git_commit and manifest.timings_seconds["total"] > 0
    assert set(manifest.ocr_row_counts) == {"val", "eval"}


def test_resume_skips_finished_scenes_and_refuses_a_different_plan(built_dataset):
    root, output_directory, manifest, arguments = built_dataset
    annotation = output_directory / "val" / "annotations" / "val_000000.json"
    before = annotation.stat().st_mtime_ns
    resumed = build_dataset(replace_argument(arguments, resume=True))
    assert annotation.stat().st_mtime_ns == before
    assert resumed.check_counts == manifest.check_counts
    assert "built 0 scenes, skipped 14" in resumed.notes[0]
    with pytest.raises(SystemExit, match="plan differs"):
        build_dataset(replace_argument(arguments, resume=True, scenes=BUILD_SCENES + 6))
    with pytest.raises(FileExistsError):
        build_dataset(arguments)


def test_one_failing_scene_is_recorded_not_fatal(built_dataset):
    root, _, _, _ = built_dataset
    plan = json.loads((root / "out" / "build_plan.json").read_text())
    plan["background_paths"] = {key: str(root / "missing.jpg") for key in plan["background_paths"]}
    (root / "fail" / "train" / "annotations").mkdir(parents=True)
    task = scene_tasks_for_plan(BuildPlan(**plan), root / "fail")[0]
    summary = run_scene_task(replace(task, scene_index=999))
    assert summary["status"] == "failed" and summary["scene_id"] == "train_000999" and summary["error"]


def test_registry_split_is_a_seeded_partition_with_exact_held_out_sizes():
    ids = [f"id_{i}" for i in range(64)]
    splits = assign_ids_by_hash_rank(ids, DEFAULT_SPLIT_FRACTIONS, seed=1, salt=2)
    assert sorted(sum(splits.values(), [])) == sorted(ids)
    assert splits == assign_ids_by_hash_rank(ids, DEFAULT_SPLIT_FRACTIONS, seed=1, salt=2)
    assert len(splits["val"]) == len(splits["eval"]) == 10
    small = assign_ids_by_hash_rank([f"f{i}" for i in range(13)], DEFAULT_SPLIT_FRACTIONS, seed=0, salt=5)
    assert [len(small[name]) for name in SPLIT_NAMES] == [9, 2, 2]
    tiny = assign_ids_by_hash_rank(["a", "b", "c"], DEFAULT_SPLIT_FRACTIONS, seed=0, salt=5)
    assert all(len(tiny[name]) == 1 for name in SPLIT_NAMES)


def test_template_split_is_stratified_so_every_family_is_held_out():
    catalog = build_template_catalog()
    family_by_id = template_family_by_id(catalog)
    pools = plan_template_pools(family_by_id, seed=0)
    assert sorted(sum(pools.values(), [])) == sorted(family_by_id)
    for family in set(family_by_id.values()):
        counts = [sum(family_by_id[template_id] == family for template_id in pools[name]) for name in SPLIT_NAMES]
        assert counts[1] >= 1 and counts[2] >= 1 and counts[0] > counts[1], (family, counts)


def test_font_pools_partition_the_whole_registry():
    pools = plan_split_pools({f"tpl_{i:03d}": "family" for i in range(24)}, ["a", "b", "c", "d"], seed=0)
    handwriting = [font for name in SPLIT_NAMES for font in pools[name].handwriting_font_ids]
    assert len(handwriting) == len(set(handwriting)) >= 24
    assert np.isclose(len(pools["train"].handwriting_font_ids) / len(handwriting), 0.7, atol=0.06)


def replace_argument(arguments, **changes):
    """A copy of parsed CLI arguments with some values changed."""
    copied = parse_arguments([])
    copied.__dict__.update({**arguments.__dict__, **changes})
    return copied


def test_stream_reproduces_the_dataset_builders_scenes(built_dataset, tmp_path):
    """One composition path: a stream with the build's seed and backgrounds yields the builder's files, byte for byte."""
    root, output_directory, manifest, _ = built_dataset
    stream = SyntheticSceneStream(manifest.seed, split="val", backgrounds=root / "backgrounds")
    for scene_index in range(manifest.scene_counts["val"]):
        composed = stream.scene(scene_index)
        scene_id = composed.label.scene_id
        assert json.dumps(composed.label_record()) == (output_directory / "val" / "annotations" / f"{scene_id}.json").read_text()
        composed.write_jpeg(tmp_path / f"{scene_id}.jpg")
        assert (tmp_path / f"{scene_id}.jpg").read_bytes() == (output_directory / "val" / "images" / f"{scene_id}.jpg").read_bytes()
