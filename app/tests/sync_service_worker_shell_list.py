"""Keep sw.js's APP_SHELL_PATHS equal to the files app/ actually ships.

    python3 app/tests/sync_service_worker_shell_list.py          # rewrite the list in sw.js
    python3 app/tests/sync_service_worker_shell_list.py --check  # exit 1 if it is stale

The service worker precaches this list so the app works with no network after the
first visit; a module missing from it would load online and 404 offline. Shipped files
are index.html plus everything under js/, styles/, fonts/ and models/ (tests/ is not shipped).
"""

import argparse
import re
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_DIRECTORIES = ("styles", "fonts", "js", "models")
SHIPPED_SUFFIXES = {".js", ".css", ".woff2", ".onnx"}
LIST_PATTERN = re.compile(r"const APP_SHELL_PATHS = \[\n(.*?)\n\];", re.DOTALL)


def list_shipped_paths() -> list[str]:
    """`./`-relative paths of every shipped file, in a stable order."""
    shipped_paths = ["./", "./index.html"]
    for directory_name in SHIPPED_DIRECTORIES:
        for file_path in sorted((APP_ROOT / directory_name).rglob("*")):
            if file_path.is_file() and file_path.suffix in SHIPPED_SUFFIXES:
                shipped_paths.append("./" + file_path.relative_to(APP_ROOT).as_posix())
    return shipped_paths


def render_list_body(shipped_paths: list[str]) -> str:
    """The lines between the brackets, one quoted path per line."""
    return "\n".join(f'  "{path}",' for path in shipped_paths)


def main() -> None:
    """Rewrite or check the list."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    service_worker_path = APP_ROOT / "sw.js"
    source = service_worker_path.read_text()
    match = LIST_PATTERN.search(source)
    if match is None:
        raise SystemExit("could not find `const APP_SHELL_PATHS = [...]` in sw.js")
    expected_body = render_list_body(list_shipped_paths())
    if arguments.check:
        if match.group(1) != expected_body:
            print("sw.js APP_SHELL_PATHS is stale; run app/tests/sync_service_worker_shell_list.py")
            sys.exit(1)
        print("sw.js APP_SHELL_PATHS matches the shipped files")
        return
    service_worker_path.write_text(source[: match.start(1)] + expected_body + source[match.end(1) :])
    print(f"wrote {len(list_shipped_paths())} paths into sw.js")


if __name__ == "__main__":
    main()
