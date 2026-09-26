"""Dump synthetic scenes as raw BGR bytes so the Node parity harness sees cv2's exact pixels.

    <venv>/bin/python app/tests/parity/dump_scene_pixels.py --split eval eval_000012 eval_000270 ...

Writes `<output>/<scene_id>.bgr` (row-major uint8, H x W x 3) and `<scene_id>.json`
({"width", "height"}). Decoding with cv2 here removes JPEG-decoder differences from a
Python-vs-JS comparison; the browser regression test covers the browser's own decode.
"""

import argparse
import json
from pathlib import Path

import cv2

SYNTHETIC_DATASET_ROOT = Path("/Volumes/vega/datasets/check-transcriber/synth/v1")
DEFAULT_OUTPUT_DIRECTORY = Path("/tmp/cts-parity")


def main() -> None:
    """Dump each requested scene."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="eval")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("scene_ids", nargs="+")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    for scene_id in arguments.scene_ids:
        image_bgr = cv2.imread(str(SYNTHETIC_DATASET_ROOT / arguments.split / "images" / f"{scene_id}.jpg"), cv2.IMREAD_COLOR)
        (arguments.output / f"{scene_id}.bgr").write_bytes(image_bgr.tobytes())
        (arguments.output / f"{scene_id}.json").write_text(json.dumps({"width": image_bgr.shape[1], "height": image_bgr.shape[0]}))
        print(scene_id, image_bgr.shape)


if __name__ == "__main__":
    main()
