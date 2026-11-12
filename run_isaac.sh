#!/bin/bash
# Run a standalone Python script inside Isaac Sim 6.2 (headless, Docker; the Ubuntu 26.04 host is not supported natively).
# Usage: ./run_isaac.sh script.py [args...]   (script path relative to ~/robot-endo, mounted at /work)
# Cache volumes keep compiled shaders between runs (the first run compiles for minutes).
set -u
C=~/docker/isaac-sim
exec docker run --rm --gpus all --network=host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ALLOW_ROOT=1 \
  -v $C/cache/main:/isaac-sim/.cache:rw \
  -v $C/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \
  -v $C/cache/kit:/isaac-sim/kit/cache:rw \
  -v $C/cache/pkg:/isaac-sim/.local/share/ov/pkg:rw \
  -v $C/cache/hub:/isaac-sim/.cache/ov/hub:rw \
  -v $C/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
  -v $C/config:/isaac-sim/.nvidia-omniverse/config:rw \
  -v $C/data:/isaac-sim/.local/share/ov/data:rw \
  -v ~/robot-endo:/work:rw \
  -v ~/medai-colon/outputs/polyp3d/web:/assets/polyp3d:ro \
  -v ~/colon-db/map:/assets/map:ro \
  --entrypoint /isaac-sim/python.sh nvcr.io/nvidia/isaac-sim:6.2.0.1 "/work/$1" "${@:2}"
