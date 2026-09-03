#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${1:-configs/local/binding_gpu_readiness.json}"
OUTPUT_PATH="${2:-audit/results/binding_gpu_readiness_remote.json}"
PYTHON_BIN="${BINDING_PYTHON_BIN:-${REPO_ROOT}/.venv-binding-gpu/bin/python}"

cd "${REPO_ROOT}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "G0 requires Linux." >&2
  exit 2
fi

if grep -qi microsoft /proc/version 2>/dev/null; then
  echo "G0 rejects WSL because ManiSkill GPU simulation is unsupported there." >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable." >&2
  exit 2
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing host-specific config: ${CONFIG_PATH}" >&2
  echo "Copy the fail-closed template into configs/local and review it first." >&2
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing isolated Python environment: ${PYTHON_BIN}" >&2
  echo "Run bootstrap_binding_env.sh only after host and cost approval." >&2
  exit 2
fi

nvidia-smi
"${PYTHON_BIN}" scripts/check_binding_gpu_readiness.py \
  --config "${CONFIG_PATH}" \
  --workspace "${REPO_ROOT}" \
  --output "${OUTPUT_PATH}"
