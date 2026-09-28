#!/usr/bin/env bash
set -euo pipefail

: "${SE3_STAGE:?Set SE3_STAGE to the staged sequence root}"
: "${SE3_OUT:?Set SE3_OUT to the precomputed temporal output root}"

export SE3_HOST="${SE3_HOST:-0.0.0.0}"
export SE3_PORT="${SE3_PORT:-8142}"
export SE3_GS_RENDER_URL="${SE3_GS_RENDER_URL:-http://127.0.0.1:8110/render}"
export SE3_CURRENT_EWA="${SE3_CURRENT_EWA:-1}"
export SE3_RIG_DENSE_MULTIVIEW="${SE3_RIG_DENSE_MULTIVIEW:-1}"
export SE3_RIG_DENSE_MULTIVIEW_STEP="${SE3_RIG_DENSE_MULTIVIEW_STEP:-2}"
export SE3_DIRECTIONAL_BACKGROUND="${SE3_DIRECTIONAL_BACKGROUND:-1}"
export SE3_INFORMATION_DOMAIN="${SE3_INFORMATION_DOMAIN:-1}"
export SE3_AI_INPAINT="${SE3_AI_INPAINT:-0}"
export SE3_AI_INPAINT_URL="${SE3_AI_INPAINT_URL:-http://127.0.0.1:8112/inpaint}"

python viewer/se3_viewer_server.py
