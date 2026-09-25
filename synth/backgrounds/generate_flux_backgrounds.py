"""Generate empty household-surface background photos with local FLUX (mflux) for the check compositor.

Run with the flux-image-gen skill's venv, which has mflux and the 4-bit schnell weights:
    ~/.claude/skills/flux-image-gen/.venv/bin/python -m synth.backgrounds.generate_flux_backgrounds \
        --count 300 --out /Volumes/vega/datasets/check-transcriber/backgrounds/flux

- The model is loaded ONCE and reused for every image (the CLI reloads 9 GB per image, which cost 7 min each).
- One process, one image at a time (16 GB machine; the skill's memory guardrail), MLX cache capped at 2 GB.
- Resumable: an image whose file already exists is skipped, so a killed run just restarts.
- Deterministic: image i uses seed BASE_SEED + i and a prompt drawn from random.Random(BASE_SEED + i).
- Every image is appended to manifest.jsonl with its prompt, seed, size and render time.
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
from mflux.models.common.config import ModelConfig
from mflux.models.flux.variants.txt2img.flux import Flux1

from synth.backgrounds.background_prompts import build_background_prompt

logger = logging.getLogger(__name__)

FLUX_SKILL_DIR = Path.home() / ".claude/skills/flux-image-gen"
MODEL_DIR = FLUX_SKILL_DIR / "models/schnell-4bit"
BASE_SEED = 20260925
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 768
SCHNELL_STEPS = 4
SCHNELL_GUIDANCE = 3.5  # mflux CLI default; schnell ignores guidance but the Config wants a value
MLX_CACHE_LIMIT_BYTES = 2 * 1000**3


def load_schnell_model() -> Flux1:
    """Load the pre-quantized 4-bit FLUX.1-schnell weights the same way the mflux CLI does."""
    mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES)
    return Flux1(model_config=ModelConfig.from_name(model_name=str(MODEL_DIR), base_model="schnell"))


def generate_one_background(flux: Flux1, index: int, output_dir: Path, width: int, height: int) -> Path | None:
    """Render background number `index` into output_dir; returns the path, or None if it already existed."""
    seed = BASE_SEED + index
    background_prompt = build_background_prompt(random.Random(seed))
    output_path = output_dir / f"bg_{index:04d}__{background_prompt.slug}__seed={seed}.png"
    if output_path.exists():
        return None
    started = time.time()
    generated_image = flux.generate_image(
        seed=seed, prompt=background_prompt.prompt, num_inference_steps=SCHNELL_STEPS,
        width=width, height=height, guidance=SCHNELL_GUIDANCE,
    )
    generated_image.save(path=str(output_path))
    with open(output_dir / "manifest.jsonl", "a") as manifest:
        manifest.write(json.dumps({
            "file": output_path.name, "index": index, "seed": seed, "slug": background_prompt.slug,
            "prompt": background_prompt.prompt, "width": width, "height": height,
            "seconds": round(time.time() - started, 1),
        }) + "\n")
    logger.info("[%d] %s in %.0fs", index, output_path.name, time.time() - started)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=IMAGE_WIDTH, help="multiple of 16; small sizes for prompt smoke tests")
    parser.add_argument("--height", type=int, default=IMAGE_HEIGHT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    args.out.mkdir(parents=True, exist_ok=True)
    flux = load_schnell_model()
    logger.info("model loaded")
    for index in range(args.start, args.start + args.count):
        generate_one_background(flux, index, args.out, args.width, args.height)
    return 0


if __name__ == "__main__":
    sys.exit(main())
