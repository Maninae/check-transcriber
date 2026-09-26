"""Background surfaces for scenes.

- `loader.py`: every JPEG/PNG under a directory tree (default `paths.BACKGROUND_DIR`,
  with `flux/` for generated surfaces and `photos/` for real ones). Background ids are
  paths relative to that root, so splits stay stable as files are added.
- `procedural.py`: synthetic fabric for tests and for runs with no real backgrounds yet.
- A FLUX generation script is expected to be dropped in here later; it only needs to
  write image files under the background root.
"""
