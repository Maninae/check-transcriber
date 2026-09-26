"""Compose rendered checks onto backgrounds as phone-photo scenes with exact labels, on demand.

Entry points: `on_demand.compose_scene_on_demand` (one scene from a seed, in memory),
`on_demand.SyntheticSceneStream` (lazy scenes of one split), `compose_scene.compose_scene`
(the coordinator over given checks and a background). Module map: CLAUDE.md.

`GENERATOR_VERSION` names the whole generator (check rendering, backgrounds, composition): bump it
whenever the same seed would produce different scenes. Dataset manifests and on-demand labels record it.
"""

GENERATOR_VERSION = "0.5.0"
