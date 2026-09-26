# experiments

Research behind the app's on-device detector and field readers. Each area has its own README with the results, the recommendation and the browser notes.

- `detection/`: finding every check in a phone photo as an exact quadrilateral (classical OpenCV baseline, learned detectors, corner refinement, the hybrid the app ships). Start with `detection/REPORT.md`.
- `field_reading/`: locating the fields on a straightened check and reading them (printed and handwritten), plus the gating rules the app uses to decide what to fill. Start with `field_reading/README.md`.

Training and evaluation scenes come from `scene_composer` (on demand) or `dataset_builder` (fixed sets); datasets and weights live outside the repo.
