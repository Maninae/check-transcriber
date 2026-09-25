"""Synthetic check dataset generator for Check Transcriber.

Three stages, each its own module: render one flat check (`render_check`), composite
several checks into a phone-photo-like scene (`compose_scene`), and build split datasets
(`build_dataset`, the CLI). See CLAUDE.md for the module map.
"""

GENERATOR_VERSION = "0.2.0"
