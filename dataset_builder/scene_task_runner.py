"""Run one scene task inside a worker: skip it if already built, and never let one failure kill the build.

A long build (thousands of scenes) must survive a crash, a Ctrl-C or one bad font/background:
- Resume: a scene whose annotation JSON exists is complete (the worker writes it last, by rename),
  so it is summarized from disk instead of regenerated.
- Failure: any exception is caught per scene and returned with the scene id and traceback;
  the coordinator logs it to `failures.jsonl` and carries on. This is the deliberate
  per-item boundary around known-fallible work, not a swallow: every failure is recorded.
"""

import json
import time
import traceback
from pathlib import Path

from dataset_builder.scene_worker import SceneTask, generate_scene, scene_id_for_task


def summary_from_existing_annotation(task: SceneTask, annotation_path: Path) -> dict:
    """The same summary `generate_scene` returns, rebuilt from a finished scene's annotation."""
    scene_label = json.loads(annotation_path.read_text())
    return {"split": task.split_name, "scene_id": scene_label["scene_id"], "check_count": len(scene_label["checks"]),
            "framing_regime": task.framing_regime,
            "template_ids": sorted({check["template_id"] for check in scene_label["checks"]}),
            "background_id": scene_label["background_id"]}


def run_scene_task(task: SceneTask) -> dict:
    """Generate one scene (or reuse it); returns a summary with `status` in {built, skipped, failed}."""
    scene_id = scene_id_for_task(task)
    annotation_path = Path(task.split_directory) / "annotations" / f"{scene_id}.json"
    if annotation_path.exists():
        return {**summary_from_existing_annotation(task, annotation_path), "status": "skipped", "seconds": 0.0}
    start_time = time.perf_counter()
    try:
        summary = generate_scene(task)
    except Exception as error:  # noqa: BLE001 - per-scene boundary, recorded by the coordinator
        return {"split": task.split_name, "scene_id": scene_id, "status": "failed", "check_count": 0,
                "framing_regime": task.framing_regime,
                "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
                "seconds": time.perf_counter() - start_time}
    return {**summary, "status": "built", "seconds": time.perf_counter() - start_time}
