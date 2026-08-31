#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
native_prefix="$repo_root/.native"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "No project-local OpenMP bootstrap is needed on $(uname -s)."
  exit 0
fi

if [[ -f "$native_prefix/lib/libomp.dylib" ]]; then
  echo "Project-local OpenMP is already available at $native_prefix/lib/libomp.dylib"
  exit 0
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required to install project-local llvm-openmp on macOS." >&2
  exit 1
fi

conda create --prefix "$native_prefix" --yes -c conda-forge llvm-openmp
test -f "$native_prefix/lib/libomp.dylib"
echo "Installed project-local OpenMP at $native_prefix/lib/libomp.dylib"
