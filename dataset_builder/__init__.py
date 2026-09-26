"""Fixed synthetic datasets as one consumer of the scene composer: splits, files, exports, OCR manifest, QA.

Entry point: `python -m dataset_builder.build_dataset`. Every scene comes from
`scene_composer.on_demand.compose_scene_on_demand`; this package adds planning, parallelism,
resume and the file formats. Module map: CLAUDE.md.
"""
