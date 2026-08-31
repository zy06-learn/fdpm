#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

if [[ "$(uname -s)" == "Darwin" ]]; then
  project_openmp="$repo_root/.native/lib/libomp.dylib"
  if [[ ! -f "$project_openmp" ]]; then
    echo "Missing project-local OpenMP runtime required by LightGBM."
    echo "Run: ./scripts/bootstrap_macos.sh"
    exit 1
  fi
  export DYLD_LIBRARY_PATH="$repo_root/.native/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

usage() {
  echo "Usage: $0 {test|smoke|tuning-smoke PATH|tune PATH [RUN_DIR]|validate-data PATH|formal PATH}"
}

command_name="${1:-}"
case "$command_name" in
  test)
    uv run pytest
    uv run ruff check .
    ;;
  smoke)
    uv run flight-delay-milp run --config configs/smoke.yaml
    ;;
  tuning-smoke)
    test "$#" -eq 2 || { usage; exit 2; }
    uv run flight-delay-milp tune --config configs/tuning_smoke.yaml --input "$2"
    ;;
  tune)
    if [[ "$#" -eq 2 ]]; then
      uv run flight-delay-milp tune --config configs/tuning.yaml --input "$2"
    elif [[ "$#" -eq 3 ]]; then
      uv run flight-delay-milp tune --config configs/tuning.yaml --input "$2" --resume-run "$3"
    else
      usage
      exit 2
    fi
    ;;
  validate-data)
    test "$#" -eq 2 || { usage; exit 2; }
    uv run flight-delay-milp validate-data --input "$2" --config configs/paper.yaml
    ;;
  formal)
    test "$#" -eq 2 || { usage; exit 2; }
    uv run flight-delay-milp run --config configs/paper.yaml --input "$2"
    ;;
  *)
    usage
    exit 2
    ;;
esac
