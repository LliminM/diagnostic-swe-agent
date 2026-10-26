#!/usr/bin/env bash
# Repair one local pydicom SWE-bench image whose legacy nose-style tests fail on pytest 8.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <existing-pydicom-swebench-image>" >&2
  exit 2
fi

image=$1
docker image inspect "$image" >/dev/null

current_version=$(docker run --rm --network none --entrypoint /bin/bash "$image" -lc \
  'source /opt/miniconda3/bin/activate testbed; python -m pytest --version')
if [[ ! $current_version =~ ^pytest[[:space:]]+8\. ]]; then
  echo "Expected pytest 8 in the source image; found: $current_version" >&2
  exit 1
fi

safe_name=${image//[^a-zA-Z0-9]/-}
backup="local/${safe_name}-before-pytest7:$(date +%Y%m%d%H%M%S)"
fixed="local/${safe_name}-pytest7:$(date +%Y%m%d%H%M%S)"
original_user=$(docker image inspect --format '{{.Config.User}}' "$image")
original_user=${original_user:-root}
docker tag "$image" "$backup"

docker build --network=host --build-arg "BASE_IMAGE=$backup" \
  --build-arg "FINAL_USER=$original_user" -t "$fixed" - <<'DOCKERFILE'
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}
USER root
RUN /opt/miniconda3/envs/testbed/bin/python -m pip install --no-cache-dir 'pytest==7.4.4'
ARG FINAL_USER=root
USER ${FINAL_USER}
DOCKERFILE

# Tag the canonical image only after the fixed image passes the legacy tests.
docker run --rm --network none --entrypoint /bin/bash "$fixed" -lc \
  'set -e; source /opt/miniconda3/bin/activate testbed; cd /testbed; python -c "import pytest; assert pytest.__version__ == '\''7.4.4'\''"; python -m pytest -q pydicom/tests/test_valuerep.py'
docker tag "$fixed" "$image"
echo "Repaired: $image"
echo "Original backup: $backup"
