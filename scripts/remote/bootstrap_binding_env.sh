#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${1:-configs/local/binding_gpu_readiness.json}"
VENV_PATH="${REPO_ROOT}/.venv-binding-gpu"

cd "${REPO_ROOT}"

if [[ "$(uname -s)" != "Linux" ]] || grep -qi microsoft /proc/version 2>/dev/null; then
  echo "The binding GPU environment requires native Linux." >&2
  exit 2
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing reviewed host-specific config: ${CONFIG_PATH}" >&2
  exit 2
fi

python3.11 - "${CONFIG_PATH}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
required = {
    "approved_cost_amount": config.get("approved_cost_amount") is not None,
    "gpu_hourly_price": isinstance(config.get("gpu_hourly_price"), (int, float))
    and not isinstance(config.get("gpu_hourly_price"), bool),
    "cost_authorized": config.get("cost_authorized") is True,
    "remote_host_identity": all(
        isinstance(config.get("remote_host", {}).get(field), str)
        and bool(config["remote_host"][field].strip())
        for field in (
            "hostname",
            "provider",
            "region",
            "instance_type",
            "expected_gpu_model",
        )
    ),
    "remote_host_approved": config.get("remote_host_approved") is True,
}
failed = [name for name, passed in required.items() if not passed]
if failed:
    raise SystemExit("Host/cost config is not approved: " + ", ".join(failed))
PY

: "${BINDING_TORCH_SPEC:?Set an exact value such as torch==X.Y.Z after inspecting the driver.}"
: "${BINDING_TORCH_INDEX_URL:?Set the official CUDA PyTorch wheel index selected for the driver.}"

python3.11 -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip
"${VENV_PATH}/bin/python" -m pip install \
  --index-url "${BINDING_TORCH_INDEX_URL}" \
  "${BINDING_TORCH_SPEC}"
"${VENV_PATH}/bin/python" -m pip install -r requirements/binding-gpu.txt

"${VENV_PATH}/bin/python" -m pytest tests/test_binding_bench.py -q
"${VENV_PATH}/bin/python" -m pip freeze

echo "Environment prepared. Re-run bash scripts/remote/g0_probe.sh and archive its output."
