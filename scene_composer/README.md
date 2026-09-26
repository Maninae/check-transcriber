# scene_composer

Composes fake checks onto a background as a phone photo of several checks on a bedsheet: layout, framing, paper curl and folds, one room light with cast shadows, a phone camera pipeline and JPEG. Every label (corners, deformed outline, orientation, field quads, visibility) is exact, because it is computed by transforming points through the same maps that warp the pixels. Module map and invariants: [CLAUDE.md](CLAUDE.md).

## On demand

A scene is addressed by `(seed, split, scene_index)`, composed in memory, and never written unless you ask:

```python
from scene_composer.on_demand import SyntheticSceneStream, compose_scene_on_demand

scene = compose_scene_on_demand(7)                   # train split, scene 0, backgrounds from the data drive
scene.photo_rgb                                      # (H, W, 3) uint8
scene.label_record()                                 # full label, same schema as a dataset annotation

for scene in SyntheticSceneStream(1, split="eval"):  # endless, lazy, eval pools only
    ...
```

`SyntheticSceneStream(seed, split)` scene i is the dataset builder's `<split>_<i:06d>` for the same seed and backgrounds, so an on-demand training loop and a fixed dataset see the same distribution and the same held-out pools.

```
python -m scene_composer.generate_one --seed 7 --out /tmp/x.jpg --labels /tmp/x.json
python -m scene_composer.generate_one --seed 7 --split eval --scene-index 3 --out /tmp/x.jpg --labels /tmp/x.json
```

## Framing regimes

How close the phone is. `wide` is v1 (1-8 checks, sometimes 12, at 70-95% fill); `close` is 2-6 checks (mostly 5-6) filling 85-98% of the frame with small or zero gaps, slight overlaps and a check cut by the frame edge 40% of the time; `single` is one check filling 70-100% at any rotation, sometimes with an edge or corner cut off, cropped so only a thin band of surface shows, or shot steeply so the far edge is clearly shorter.

```python
compose_scene_on_demand(7, "eval", 3, framing_regime="single")          # one regime, explicitly
SyntheticSceneStream(7, split="train", framing_regime_mix={"close": 1})  # per scene from a mix (default 40/45/15, as builds)
```

Labels record the regime in `effects.framing.framing_regime` (absent means wide, as in every v1 label).

## Harmonization (optional)

`harmonize=True` (or `--harmonize` on the CLIs) runs PCT-Net on each pasted check's albedo, ~1 s extra per scene. It needs torch and PCT-Net cloned to `/Volumes/vega/ai-models/harmonizer/pctnet` (`git clone https://github.com/rakutentech/PCT-Net-Image-Harmonization.git pctnet`); its CNN weights ship inside that repo.

- PCT-Net CNN (Guerreiro et al., WACV 2023): MPL-2.0 code and bundled weights, used unmodified from its own clone; weights trained on iHarmony4 (COCO, Flickr, MIT-Adobe FiveK, day2night).
- Harmonizer (Ke et al., ECCV 2022) was rejected: CC BY-NC-SA 4.0, and these outputs train a model that ships in a public tool.
- `augraphy` is not used (last release Dec 2023; hard-requires full `opencv-python`, numba, scikit-learn, matplotlib). Its camera and paper effects are implemented here (`lighting/`, `camera/`).
