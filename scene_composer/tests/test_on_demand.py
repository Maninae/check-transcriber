"""On-demand composition: deterministic per (seed, split, index), lazy streams stay in their split, the CLI writes valid files."""

import json

import cv2
import numpy as np
import pytest

from scene_composer.framing_regime_mix import DEFAULT_FRAMING_REGIME_MIX, framing_regime_for_scene
from scene_composer.generate_one import main as generate_one_main
from scene_composer.on_demand import SyntheticSceneStream, compose_scene_on_demand
from scene_composer.scene_ingredient_pools import library_ingredient_pools
from synthetic_backgrounds.procedural import write_procedural_backgrounds

SEED = 5


@pytest.fixture(scope="module")
def background_root(tmp_path_factory):
    """Nine procedural fabrics under an accepted subfolder, enough for every split to get some."""
    root = tmp_path_factory.mktemp("backgrounds")
    write_procedural_backgrounds(root / "photos", 9, seed=2)
    return root


def check_ingredients(label: dict) -> list[dict]:
    """Per check: template, fonts, payee and bank as written into the label."""
    return [{"template_ids": check["template_id"],
             "handwriting_font_ids": check["canonical"]["handwriting_font_id"],
             "signature_font_ids": check["canonical"]["signature_font_id"],
             "payee_names": check["canonical"]["payee_canonical"],
             "bank_names": [field["text"] for field in check["fields"] if field["field_name"] == "bank_name"]}
            for check in label["checks"]]


def test_same_seed_gives_the_same_bytes(background_root):
    first = compose_scene_on_demand(SEED, "train", 1, backgrounds=background_root)
    again = compose_scene_on_demand(SEED, "train", 1, backgrounds=background_root)
    other = compose_scene_on_demand(SEED, "train", 2, backgrounds=background_root)
    assert first.photo_rgb.dtype == np.uint8 and first.photo_rgb.shape == (first.label.image_height, first.label.image_width, 3)
    assert np.array_equal(first.photo_rgb, again.photo_rgb)
    assert json.dumps(first.label_record()) == json.dumps(again.label_record())
    assert first.label.scene_id == "train_000001"
    assert first.photo_rgb.shape != other.photo_rgb.shape or not np.array_equal(first.photo_rgb, other.photo_rgb)


def test_stream_honours_split_pools(background_root):
    eval_pools = library_ingredient_pools(SEED, "eval", background_root)
    train_pools = library_ingredient_pools(SEED, "train", background_root)
    train_background_ids = {background.background_id for background in train_pools.backgrounds}
    assert train_background_ids.isdisjoint(background.background_id for background in eval_pools.backgrounds)
    stream = SyntheticSceneStream(SEED, split="eval", backgrounds=background_root, scene_count=3)
    scenes = list(stream)
    assert [scene.label.scene_id for scene in scenes] == ["eval_000000", "eval_000001", "eval_000002"]
    for scene in scenes:
        label = scene.label_record()
        assert label["background_id"] in {background.background_id for background in eval_pools.backgrounds}
        assert label["background_id"] not in train_background_ids
        for ingredients in check_ingredients(label):
            for kind, used in ingredients.items():
                for value in used if isinstance(used, list) else [used]:
                    assert value in getattr(eval_pools, kind) and value not in getattr(train_pools, kind), (kind, value)


def test_stream_is_lazy_and_repeatable(background_root):
    stream = SyntheticSceneStream(SEED, split="val", backgrounds=background_root, start_index=4, scene_count=2)
    iterator = iter(stream)
    first = next(iterator)
    assert first.label.scene_id == "val_000004"
    assert stream.scene(4).label_record() == first.label_record()
    assert [scene.label.scene_id for scene in stream] == ["val_000004", "val_000005"]
    with pytest.raises(ValueError, match="unknown split"):
        SyntheticSceneStream(SEED, split="test", backgrounds=background_root)


def test_generate_one_writes_a_valid_jpeg_and_label_json(background_root, tmp_path):
    photo_path, label_path = tmp_path / "out" / "scene.jpg", tmp_path / "out" / "scene.json"
    generate_one_main(["--seed", "7", "--backgrounds", str(background_root), "--out", str(photo_path), "--labels", str(label_path)])
    assert photo_path.read_bytes()[:3] == b"\xff\xd8\xff"
    photo = cv2.imread(str(photo_path))
    label = json.loads(label_path.read_text())
    assert photo.shape == (label["image_height"], label["image_width"], 3)
    assert label["provenance"] == {**label["provenance"], "seed": 7, "split": "train", "scene_index": 0}
    assert label["provenance"]["generator_version"]
    assert label["checks"] and all(len(check["corners"]) == 4 for check in label["checks"])
    regime = framing_regime_for_scene(7, "train", 0, DEFAULT_FRAMING_REGIME_MIX)
    assert label["provenance"]["framing_regime"] == regime
    expected = json.loads(json.dumps(compose_scene_on_demand(7, backgrounds=background_root, framing_regime=regime).label_record()))
    assert {key: value for key, value in label.items() if key != "provenance"} == expected
