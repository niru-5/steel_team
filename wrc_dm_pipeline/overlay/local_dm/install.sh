#!/usr/bin/env bash
# One-time setup for the WRC B601-DM port. Safe to re-run. See local_dm/README.md.
# Checks both Python environments, generates the DM URDF and fetches the detector's
# text encoder. It does not create the conda env or the teleoperation venv for you;
# it tells you the command when one is missing.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TELEOP="${STEEL_TEAM_TELEOP_DIR:-$REPO/../steel_team/teleoperation}"
cd "$REPO"
ok()   { echo "  [ok]   $*"; }
todo() { echo "  [TODO] $*"; FAIL=1; }
FAIL=0

echo "1. WRC conda env (wrc-demo)"
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
WRC_PY="$CONDA_BASE/envs/wrc-demo/bin/python"
if [ -x "$WRC_PY" ]; then
  ok "$WRC_PY"
  if [ "$(uname)" = "Darwin" ] && conda list -n wrc-demo 2>/dev/null | grep -q "libopenblas.*openmp"; then
    echo "  macOS: switching OpenBLAS to the pthreads build (two OpenMP copies crash torch)"
    conda install -n wrc-demo -c conda-forge -y "libopenblas=*=*pthreads*" >/dev/null
  fi
  PYTHONNOUSERSITE=1 "$WRC_PY" -s -c "import torch, ultralytics, pinocchio, anthropic, wrc_demo" \
    && ok "imports: torch, ultralytics, pinocchio, anthropic, wrc_demo" \
    || todo "imports failed in wrc-demo (re-run the pip block from environment.yml; see SETUP_REPORT.md)"
else
  todo "conda env missing: PYTHONNOUSERSITE=1 conda env create -f environment.yml"
fi

echo "2. teleoperation venv (MuJoCo, DM model)"
if [ -x "$TELEOP/.venv/bin/python" ] && [ -f "$TELEOP/assets/reBot-DevArm/Rebot_Arm_description/DM/urdf/ReBot_Arm_DM.urdf" ]; then
  ok "$TELEOP"
  if [ "$(uname)" = "Darwin" ] && [ -x "$TELEOP/.venv/bin/mjpython" ]; then
    # mjpython needs libpython next to the venv's python; uv-managed interpreters keep it elsewhere.
    LIBDIR="$("$TELEOP/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
    LIBNAME="$("$TELEOP/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("LDLIBRARY"))')"
    if [ ! -e "$TELEOP/.venv/lib/$LIBNAME" ] && [ -f "$LIBDIR/$LIBNAME" ]; then
      ln -s "$LIBDIR/$LIBNAME" "$TELEOP/.venv/lib/$LIBNAME" && echo "  macOS: linked $LIBNAME for mjpython"
    fi
  fi
else
  todo "in $TELEOP run:  ./scripts/setup_assets.sh && uv sync   (or set STEEL_TEAM_TELEOP_DIR)"
fi

echo "3. DM URDF for WRC"
if [ -x "$WRC_PY" ] && [ -f "$TELEOP/assets/reBot-DevArm/Rebot_Arm_description/DM/urdf/ReBot_Arm_DM.urdf" ]; then
  STEEL_TEAM_TELEOP_DIR="$TELEOP" "$WRC_PY" -s local_dm/make_dm_urdf.py >/dev/null && ok "local_dm/assets/ReBot_Arm_DM_wrc.urdf"
else
  todo "needs steps 1 and 2 first"
fi

echo "4. Detector weights and text encoder"
[ -f models/yoloe-26l-seg.pt ] && ok "models/yoloe-26l-seg.pt" || todo "models/yoloe-26l-seg.pt missing (it ships in the WRC repo)"
if [ ! -f models/mobileclip2_b.ts ] && [ -x "$WRC_PY" ]; then
  echo "  downloading the YOLOE text encoder (~250 MB, one time; also installs ultralytics' 'clip' package)"
  (cd models && PYTHONNOUSERSITE=1 "$WRC_PY" -s -c "
from ultralytics import YOLOE
m = YOLOE('yoloe-26l-seg.pt'); m.set_classes(['red block'], m.get_text_pe(['red block']))" >/dev/null 2>&1) || true
fi
if [ -f models/mobileclip2_b.ts ]; then
  ok "models/mobileclip2_b.ts"
  [ -e mobileclip2_b.ts ] || ln -s models/mobileclip2_b.ts mobileclip2_b.ts   # ultralytics looks in the CWD
else
  todo "text encoder not downloaded (network?). Re-run this script."
fi

echo "5. Claude API key"
if [ -n "${ANTHROPIC_API_KEY:-}" ] || grep -qs '^ANTHROPIC_API_KEY=.\+' .env; then
  ok "found (value not shown)"
else
  echo "  [info] no key: only the built-in keyword commands will work."
  echo "         echo 'ANTHROPIC_API_KEY=...' > .env && chmod 600 .env"
fi

echo
if [ "$FAIL" = 0 ]; then echo "Ready:  ./START_WRC.sh dm"; else echo "Fix the [TODO] items above, then re-run."; exit 1; fi
