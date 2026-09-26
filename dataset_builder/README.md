# dataset_builder

Builds fixed, held-out synthetic datasets of "phone photo of several checks on a bedsheet" scenes for Check Transcriber, because no public dataset has that input. Scenes come from `scene_composer` (the same function an on-demand stream uses); this package splits, parallelizes, resumes and exports. Module map and invariants: [CLAUDE.md](CLAUDE.md).

## Usage

```
python -m dataset_builder.build_dataset \
  --output /Volumes/vega/datasets/check-transcriber/synth/run-01 \
  --scenes 10000 \
  --seed 1 \
  --workers 2

# see split pools and counts without rendering anything
python -m dataset_builder.build_dataset --output DIR --scenes 5000 --plan-only

# continue an interrupted build (same arguments; finished scenes are skipped)
python -m dataset_builder.build_dataset --output DIR --scenes 5000 --resume

# OCR manifest for train too (default: val,eval)
python -m dataset_builder.build_dataset --output DIR --scenes 200 --ocr-splits train,val,eval

# no real backgrounds yet: use procedural fabric instead
python -m dataset_builder.build_dataset --output DIR --scenes 200 --procedural-backgrounds 12

# network harmonization of each pasted check (slower; blend = share of the network's capped colour shift)
python -m dataset_builder.build_dataset --output DIR --scenes 200 --harmonize --harmonize-blend 0.5

# print sheets (same as python -m synthetic_checks.print_sheets)
python -m dataset_builder.build_dataset --output DIR --print-sheets 4 --print-pool eval --seed 1

# visual QA: 12 scenes with label polygons drawn
python -m dataset_builder.qa.contact_sheet DIR --out /tmp/sheet.png

# visual QA: OCR field crops with their ground-truth text underneath
python -m dataset_builder.qa.ocr_crop_grid DIR --out /tmp/ocr_grid.png --split val

# presentation: contact sheet, full scene, 3x crop beside a real photo, flat check, print page (JPEGs under 5 MB)
python -m dataset_builder.showcase_images DIR --out /tmp/showcase --print-page PRINT_DIR/print_sheet__page=02.png
```

## Output layout

```
DIR/manifest.json                        version, git commit, command, seed, counts, split pools, timings, failures, config
DIR/build_plan.json                      the plan the build started with (checked on --resume)
DIR/failures.jsonl                       per-scene failures with id and traceback (only if any)
DIR/data.yaml, DIR/data_obb.yaml         Ultralytics configs (segmentation; oriented boxes via DIR/yolo_obb symlinks)
DIR/{train,val,eval}/images/*.jpg
DIR/{train,val,eval}/annotations/*.json  full scene labels
DIR/{train,val,eval}/labels/*.txt        YOLO segmentation (clipped polygon)
DIR/{train,val,eval}/labels_obb/*.txt    YOLO oriented box (4 corners)
DIR/{train,val,eval}/annotations_coco.json  COCO: checks (outline + 4 corner keypoints TL TR BR BL, orientation)
                                            and fields (one category per field, text in attributes)
DIR/ocr/<split>/checks/*.jpg             each check rectified from its corners to 1600 px wide, upright
DIR/ocr/<split>/fields/*.png             each field cropped from that (box + small margin)
DIR/ocr/ocr_fields__split=<split>.jsonl  one row per field: text, handwritten, status/usable, box, crop paths, fonts
```

## Split rule (contract C4)

Template ids (stratified by layout family), background ids, handwriting and signature font ids, payee names and bank names are each partitioned 70/15/15, so eval checks use stock, surfaces, hands and names train never saw. Each id is placed by its own hash, so a background pool of 50+ never moves an existing id when more backgrounds land (details in `synthetic_checks/splits.py`). OCR rows are `usable` only when the field is fully in frame, not covered by another check, and its ink is at least 14 px tall in the photo (`too_small` otherwise); the MICR band is kept as a flagged row (the app blurs it).
