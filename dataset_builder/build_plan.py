"""The build plan: what a dataset build will contain, decided before anything is rendered.

The plan fixes the split pools (contract C4), scene counts, backgrounds and scene config. It is
printed by `--plan-only`, and written as `build_plan.json` before the first scene so a resumed
build (`--resume`) can refuse to continue with different pools or counts.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from scene_composer.scene_config import SceneConfig
from synthetic_backgrounds.loader import list_background_sources
from synthetic_backgrounds.procedural import write_procedural_backgrounds
from synthetic_checks.check_templates import build_template_catalog, template_family_by_id
from synthetic_checks.splits import DEFAULT_SPLIT_FRACTIONS, SPLIT_NAMES, plan_split_pools, scene_counts_per_split
from synthetic_data_paths import DATA_ROOT

BUILD_PLAN_FILENAME = "build_plan.json"
PROCEDURAL_BACKGROUND_DIR = DATA_ROOT / "procedural-backgrounds"


@dataclass
class BuildPlan:
    """Everything that determines a build's content (not its speed: workers are excluded)."""

    seed: int
    template_count: int
    scene_counts: dict[str, int]
    split_fractions: dict[str, float]
    split_pools: dict[str, dict[str, list[str]]]
    background_paths: dict[str, str]    # background id -> file path
    background_root: str
    scene_config: dict
    ocr_splits: list[str]

    def to_dict(self) -> dict:
        """Plain-JSON form (tuples become lists, as they read back)."""
        return json.loads(json.dumps(asdict(self)))


def resolve_backgrounds(background_root: Path, procedural_count: int, seed: int) -> tuple[Path, dict[str, str]]:
    """Background id -> path; procedural fabrics are generated (cached on the data drive) when requested."""
    if procedural_count:
        background_root = PROCEDURAL_BACKGROUND_DIR / f"seed_{seed}"
        paths = write_procedural_backgrounds(background_root, procedural_count, seed)
        return background_root, {path.name: str(path) for path in paths}
    sources = list_background_sources(background_root)
    return background_root, {source.background_id: str(source.file_path) for source in sources}


def make_build_plan(seed: int, scene_total: int, template_count: int, background_root: Path, procedural_count: int,
                    scene_config: SceneConfig, ocr_splits: list[str]) -> BuildPlan:
    """Resolve backgrounds and partition every id kind into split pools."""
    background_root, background_paths = resolve_backgrounds(background_root, procedural_count, seed)
    if len(background_paths) < len(SPLIT_NAMES):
        raise SystemExit(f"need at least 3 backgrounds under {background_root}, found {len(background_paths)}; "
                         "add photos, wait for the FLUX batch, or pass --procedural-backgrounds N")
    pools = plan_split_pools(template_family_by_id(build_template_catalog(template_count)), sorted(background_paths), seed)
    return BuildPlan(
        seed=seed, template_count=template_count,
        scene_counts=scene_counts_per_split(scene_total, DEFAULT_SPLIT_FRACTIONS),
        split_fractions=DEFAULT_SPLIT_FRACTIONS,
        split_pools={split_name: pools[split_name].to_dict() for split_name in SPLIT_NAMES},
        background_paths=background_paths, background_root=str(background_root),
        scene_config=json.loads(json.dumps(asdict(scene_config))), ocr_splits=ocr_splits,
    )


def scene_config_from_plan(plan: BuildPlan) -> SceneConfig:
    """Rebuild the SceneConfig from its JSON form (lists back to the tuples the dataclass declares)."""
    return SceneConfig(**{key: tuple(value) if isinstance(value, list) else value for key, value in plan.scene_config.items()})


def format_build_plan(plan: BuildPlan) -> str:
    """Human-readable plan: per split, scene count and pool sizes, then every pool's ids."""
    lines = [f"seed {plan.seed}, {plan.template_count} templates, backgrounds from {plan.background_root}",
             f"OCR manifest for: {', '.join(plan.ocr_splits) or 'no splits'}", ""]
    kinds = list(plan.split_pools[SPLIT_NAMES[0]])
    lines.append(f"{'split':<6} {'scenes':>7} " + " ".join(f"{kind.removesuffix('_ids'):>16}" for kind in kinds))
    for split_name in SPLIT_NAMES:
        pools = plan.split_pools[split_name]
        lines.append(f"{split_name:<6} {plan.scene_counts[split_name]:>7} " + " ".join(f"{len(pools[kind]):>16}" for kind in kinds))
    for split_name in SPLIT_NAMES:
        lines.append(f"\n[{split_name}]")
        lines += [f"  {kind}: {', '.join(ids)}" for kind, ids in plan.split_pools[split_name].items()]
    return "\n".join(lines)


def write_build_plan(plan: BuildPlan, output_directory: Path) -> None:
    """Record the plan before rendering starts."""
    (output_directory / BUILD_PLAN_FILENAME).write_text(json.dumps(plan.to_dict(), indent=2) + "\n")


def require_matching_build_plan(plan: BuildPlan, output_directory: Path) -> None:
    """A resumed build must have exactly the plan the directory was started with."""
    plan_path = output_directory / BUILD_PLAN_FILENAME
    if not plan_path.exists():
        raise SystemExit(f"--resume needs {plan_path}; this directory was not started by build_dataset")
    recorded = json.loads(plan_path.read_text())
    differing = sorted(key for key, value in plan.to_dict().items() if recorded.get(key) != value)
    if differing:
        raise SystemExit(f"--resume refused: plan differs from the recorded one in {differing}; "
                         "use the original arguments or a new --output")
