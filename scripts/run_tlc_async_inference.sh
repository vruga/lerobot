#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_tlc_async_inference.sh [config]
#
# Available configs:
#   default          - Standard spec with atomic delivery (default)
#   network-failures - Test safety properties under network failures
#   aggregation      - Test with abstract aggregation (expensive!)
#
# Examples:
#   ./scripts/run_tlc_async_inference.sh
#   ./scripts/run_tlc_async_inference.sh network-failures
#   ./scripts/run_tlc_async_inference.sh aggregation

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  echo "error: must be run inside a git checkout (to locate repo root)" >&2
  exit 1
fi

SPEC_DIR="${REPO_ROOT}/src/lerobot/async_inference/tla"
TLA_FILE="${SPEC_DIR}/AsyncInference.tla"

# Select config based on argument
CONFIG_NAME="${1:-default}"
case "${CONFIG_NAME}" in
  default)
    CFG_FILE="${SPEC_DIR}/AsyncInference.cfg"
    ;;
  network-failures|network_failures|failures)
    CFG_FILE="${SPEC_DIR}/AsyncInference_NetworkFailures.cfg"
    ;;
  aggregation|agg)
    CFG_FILE="${SPEC_DIR}/AsyncInference_Aggregation.cfg"
    echo "WARNING: Aggregation config uses non-determinism and is expensive!"
    ;;
  *)
    echo "error: unknown config '${CONFIG_NAME}'" >&2
    echo "Available: default, network-failures, aggregation" >&2
    exit 1
    ;;
esac

if [[ ! -f "${TLA_FILE}" ]]; then
  echo "error: missing ${TLA_FILE}" >&2
  exit 1
fi
if [[ ! -f "${CFG_FILE}" ]]; then
  echo "error: missing ${CFG_FILE}" >&2
  exit 1
fi

echo "Using config: ${CFG_FILE}"

# Allow user override.
if [[ -n "${TLA_TOOLS_JAR:-}" ]]; then
  if [[ ! -f "${TLA_TOOLS_JAR}" ]]; then
    echo "error: TLA_TOOLS_JAR was set but file does not exist: ${TLA_TOOLS_JAR}" >&2
    exit 1
  fi
else
  # Try common locations.
  CANDIDATES=(
    "/opt/tla/toolbox/tla2tools.jar"
    "${REPO_ROOT}/tla2tools.jar"
    "${HOME}/tla2tools.jar"
    "${HOME}/Downloads/tla2tools.jar"
    "${HOME}/.local/share/tla2tools.jar"
    "/usr/share/java/tla2tools.jar"
    "/usr/local/share/tla2tools.jar"
    "${HOME}/TLA+Toolbox/tla2tools.jar"
    "${HOME}/.tla/tla2tools.jar"
    "${HOME}/.tlaplus/tla2tools.jar"
  )

  for c in "${CANDIDATES[@]}"; do
    if [[ -f "${c}" ]]; then
      TLA_TOOLS_JAR="${c}"
      break
    fi
  done
fi

if [[ -z "${TLA_TOOLS_JAR:-}" ]]; then
  cat >&2 <<'EOF'
error: could not find tla2tools.jar

Set TLA_TOOLS_JAR to the absolute path of your tla2tools.jar, e.g.
  TLA_TOOLS_JAR=/path/to/tla2tools.jar ./scripts/run_tlc_async_inference.sh

You can download it via the TLA+ tools / Toolbox distribution.
EOF
  exit 1
fi

TLC_WORKERS="${TLC_WORKERS:-auto}"
TLC_ARGS="${TLC_ARGS:-}"

cd "${SPEC_DIR}"
exec java -cp "${TLA_TOOLS_JAR}" tlc2.TLC \
  -config "${CFG_FILE}" \
  -workers "${TLC_WORKERS}" \
  ${TLC_ARGS} \
  "${TLA_FILE}"
