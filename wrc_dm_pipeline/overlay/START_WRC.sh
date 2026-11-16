#!/usr/bin/env bash
# START_WRC.sh — launch the verified WRC MOCK workflow on this Mac.
# (Added by the hackathon setup; not part of upstream WRC.)
#
# Everything here is mock-only: mock camera, mock detector, mock arm,
# scripted mock LLM. No hardware, no network/LLM API calls, no physical motion.
#
# Usage:
#   ./START_WRC.sh                 # interactive REPL (type: look around / go home / open the gripper /
#                                  #   pick up the red cube and put it in the box ...); empty line quits
#   ./START_WRC.sh task "go home"  # one task through the stock CLI (wrc_demo.apps.demo)
#   ./START_WRC.sh demo            # verified E2E: scripted multi-step plan + simulated grasp contact
#                                  #   (local_mock/run_mock_task.py --simulate-contact --scripted-plan)
#   ./START_WRC.sh dashboard       # WRC's live web dashboard at http://127.0.0.1:8090/ (type tasks in the
#                                  #   chat box; simulated grasp contact; localhost only)
#   ./START_WRC.sh dm              # reBot B601-DM: physics sim window + dashboard http://127.0.0.1:8090/
#                                  #   (Claude plans if ANTHROPIC_API_KEY is set / in wrc/.env) — local_dm/README.md
#                                  #   Scene: the team's real workspace (steel_team simulation/: can, cup, foam
#                                  #   bin) when that folder is found; WRC_DM_SCENE=demo forces the old demo table.
#   ./START_WRC.sh dm-task "put the can in the bin"   # one DM task, headless
#   ./START_WRC.sh dm-shuffle      # while 'dm' is running: move the sim objects to new random places
#   ./START_WRC.sh dm-trials       # randomized pick-and-place trials in the sim -> success rate
#                                  #   (--n 5, --seed 7, --no-llm for keyword commands without API calls)
#   ./START_WRC.sh dm-calibrate    # camera calibration from the printed sheet (no arguments: shows the steps)
#   ./START_WRC.sh dm-calib-test   # check the camera calibration helper against the sim's true camera
#   ./START_WRC.sh dm-measure --camera A --camera2 B --table-z 0.0   # no depth camera: what the two scene
#                                  #   cameras measure (place, height, width of each object). Does not move the arm.
#   ./START_WRC.sh dm-measure-test # self-test of that measuring on drawn scenes (no simulator, no cameras)
#                                  #   Extra options for dm / dm-task go in WRC_DM_EXTRA, e.g.
#                                  #   WRC_DM_EXTRA="--camera my_cam --camera2 my_cam2 --table-z 0.0 --known-objects my_objects.yaml"
#   ./START_WRC.sh 3d              # (older) mock RS arm mirrored in a 3D window
#   ./START_WRC.sh tests           # README's documented quick tests (expect 17 passed, 1 skipped)
#   ./START_WRC.sh tests-full      # full mock suite (last run: 381 passed, 20 failed, 5 skipped)
#   ./START_WRC.sh mcp-smoke       # launch the MCP stdio server in mock mode and exercise its tools
#   ./START_WRC.sh mcp             # run the MCP stdio server (for an MCP host to attach to)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="wrc-demo"
CALLER_PWD="$PWD"          # dm-calibrate resolves the user's file paths from where they ran it
cd "$REPO"

# Activate the conda env created from environment.yml.
CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/anaconda3)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export PYTHONNOUSERSITE=1      # never pick up ~/.local site-packages (see CLAUDE.md torch gotcha)
export YOLO_OFFLINE=True       # stop ultralytics from phoning GitHub
export WRC_ARM=mock            # MCP server: mock arm
export WRC_VIEW=0              # no OpenCV window
export WRC_STREAM=0            # no MJPEG dashboard for the MCP server
STAMP="$(date +%Y%m%d_%H%M%S)"

# Where the team's steel_team checkout lives (its teleoperation/ folder has the arm model and the
# physics sim; its simulation/ folder -- on main since 2026-11-15 -- is the real-workspace scene).
# Set STEEL_TEAM_TELEOP_DIR to choose; otherwise prefer a checkout that has simulation/.
if [ -z "${STEEL_TEAM_TELEOP_DIR:-}" ]; then
  STEEL_TEAM_TELEOP_DIR="$REPO/../steel_team/teleoperation"
  for d in "$REPO/../steel_team" "$REPO/../steel_team_main"; do
    if [ -d "$d/simulation/rebot_sim" ] && [ -x "$d/teleoperation/.venv/bin/python" ]; then
      STEEL_TEAM_TELEOP_DIR="$d/teleoperation"; break
    fi
  done
fi
export STEEL_TEAM_TELEOP_DIR

# Refuse to start a second dashboard / 3D window on ports already in use.
require_free_ports() {
  if lsof -nP -iTCP:8090 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "WRC dashboard port 8090 is already in use — a WRC dashboard is probably already running"
    echo "(another './START_WRC.sh dashboard' or './START_WRC.sh 3d'). Open http://127.0.0.1:8090/ to use it,"
    echo "or stop it first: Ctrl+C in its terminal, or  pkill -f local_mock/dashboard_mock.py"
    exit 1
  fi
  if [ "${1:-}" = "3d" ] && lsof -nP -iUDP:9870 >/dev/null 2>&1; then
    echo "3D mirror port 9870 is already in use — a WRC 3D window is probably already open."
    echo "Close that window, or stop it with:  pkill -f local_mock/wrc_mujoco_mirror.py"
    exit 1
  fi
}

mode="${1:-repl}"
case "$mode" in
  repl)
    exec python -m wrc_demo.apps.demo --arm mock --llm mock --no-view --no-serve \
      --interactive --run-dir "runs/repl_$STAMP" ;;
  task)
    shift
    exec python -m wrc_demo.apps.demo --arm mock --llm mock --no-view --no-serve \
      --run-dir "runs/task_$STAMP" --task "${*:-look around}" ;;
  demo)
    exec python local_mock/run_mock_task.py --simulate-contact --scripted-plan \
      --run-dir "runs/demo_$STAMP" ;;
  tests)
    exec python -m pytest tests/test_extrinsics_loader.py \
      tests/test_hand_eye_compensation_per_camera.py \
      tests/test_demo_yaml_llm_profile.py -q -p no:cacheprovider ;;
  tests-full)
    exec python -m pytest tests/ -q -rfEs -p no:cacheprovider ;;
  dashboard)
    # WRC's live web dashboard (chat box + narration + world model) on http://127.0.0.1:8090/
    require_free_ports
    exec python local_mock/dashboard_mock.py ;;
  dm-trials)
    # Randomized pick-and-place trials in the DM physics sim -> success rate (local_dm/sim_trials.py).
    shift
    [ -e mobileclip2_b.ts ] || [ ! -e models/mobileclip2_b.ts ] || ln -s models/mobileclip2_b.ts mobileclip2_b.ts
    exec python local_dm/sim_trials.py "$@" ;;
  dm-shuffle)
    # Move the objects of the running DM simulator to new random places.
    exec python local_dm/shuffle_sim.py ;;
  dm-measure)
    # No depth camera: what the scene cameras measure for each object. Moves nothing.
    shift
    [ -e mobileclip2_b.ts ] || [ ! -e models/mobileclip2_b.ts ] || ln -s models/mobileclip2_b.ts mobileclip2_b.ts
    exec python local_dm/measure_objects.py "$@" ;;
  dm-measure-test)
    # Self-test of the camera-outline measuring: the maths on drawn scenes, then the wiring into the pipeline.
    python local_dm/two_view_selftest.py
    exec python local_dm/camera_measure_wiring_test.py ;;
  dm-calib-test)
    # Self-test of the camera calibration helper against the sim's known camera pose.
    exec python local_dm/calibration_selftest.py ;;
  dm-calibrate)
    # Camera-to-base calibration from the printed sheet (local_dm/README.md "Camera calibration").
    # Arguments go straight to calibrate_camera.py; file paths are relative to where you ran this.
    shift
    if [ $# -eq 0 ]; then
      echo "Camera calibration (see local_dm/README.md):"
      echo "  1. print    $REPO/local_dm/calib_sheet_A4.png  (A4, 100 %), lay it flat in the camera view"
      echo "  2. measure  crosses A and B from the arm base; start from this file:"
      echo "                $0 dm-calibrate --example-config > my_calib.yaml"
      echo "  3. run      $0 dm-calibrate --camera uvc --config my_calib.yaml --write-profile my_camera"
      echo "No sheet or camera yet? Check the tool in the simulator:  $0 dm-calib-test"
      exit 0
    fi
    cd "$CALLER_PWD"
    exec python "$REPO/local_dm/calibrate_camera.py" "$@" ;;
  dm|dm-headless|dm-task)
    # reBot B601-DM: MuJoCo PHYSICS sim (bridge server) + WRC pipeline (+ Claude if ANTHROPIC_API_KEY
    # is in the environment or in wrc/.env). See local_dm/README.md.
    TELEOP="$STEEL_TEAM_TELEOP_DIR"
    PY="$TELEOP/.venv/bin/python"; MJPY="$TELEOP/.venv/bin/mjpython"
    [ -x "$PY" ] || { echo "teleoperation venv not found at $TELEOP/.venv — run 'uv sync' there (see local_dm/README.md)"; exit 1; }
    [ -f local_dm/assets/ReBot_Arm_DM_wrc.urdf ] || python local_dm/make_dm_urdf.py
    if [ "$(uname)" = "Darwin" ] && [ -x "$MJPY" ]; then
      # mjpython (the 3D window) needs libpython next to the venv's python; uv keeps it elsewhere.
      LIBDIR="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
      LIBNAME="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("LDLIBRARY"))')"
      [ -e "$TELEOP/.venv/lib/$LIBNAME" ] || [ ! -f "$LIBDIR/$LIBNAME" ] || ln -s "$LIBDIR/$LIBNAME" "$TELEOP/.venv/lib/$LIBNAME"
    fi
    [ -e mobileclip2_b.ts ] || [ ! -e models/mobileclip2_b.ts ] || ln -s models/mobileclip2_b.ts mobileclip2_b.ts
    BRIDGE_PID=""
    if lsof -nP -iTCP:9881 -sTCP:LISTEN >/dev/null 2>&1; then
      echo "DM bridge already running on port 9881 — using it."
    else
      # --scene: auto = the team's real-workspace scene when simulation/ is there, else the demo table.
      SCENE_ARGS=(--teleop-dir "$TELEOP" --scene "${WRC_DM_SCENE:-auto}")
      if [ "$mode" = "dm" ] && [ -x "$MJPY" ]; then "$MJPY" local_dm/dm_bridge_server.py "${SCENE_ARGS[@]}" &
      else "$PY" local_dm/dm_bridge_server.py --headless "${SCENE_ARGS[@]}" & fi
      BRIDGE_PID=$!
      trap '[ -n "$BRIDGE_PID" ] && kill $BRIDGE_PID 2>/dev/null' EXIT
      for _ in $(seq 1 60); do lsof -nP -iTCP:9881 -sTCP:LISTEN >/dev/null 2>&1 && break; sleep 0.5; done
      lsof -nP -iTCP:9881 -sTCP:LISTEN >/dev/null 2>&1 || { echo "DM bridge did not start"; exit 1; }
    fi
    if [ "$mode" = "dm-task" ]; then
      shift
      # shellcheck disable=SC2086  # WRC_DM_EXTRA is a list of options, split on purpose
      python local_dm/run_dm.py ${WRC_DM_EXTRA:-} --task "${*:-look around and tell me what you see}"
    else
      if lsof -nP -iTCP:8090 -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Dashboard port 8090 is already in use — stop the other WRC dashboard first (pkill -f local_dm/run_dm.py; pkill -f local_mock/dashboard_mock.py)"; exit 1
      fi
      # shellcheck disable=SC2086
      python local_dm/run_dm.py ${WRC_DM_EXTRA:-} --dashboard
    fi ;;
  3d)
    # Dashboard + MuJoCo 3D window that mirrors WRC's arm (Seeed reBot B601-RS model).
    # Uses the teleoperation folder's venv (it has mujoco); setup: steel_team/teleoperation/README.md
    MJPY="$REPO/../steel_team/teleoperation/.venv/bin/mjpython"
    if [ ! -x "$MJPY" ]; then
      echo "mjpython not found at $MJPY — run 'uv sync' in steel_team/teleoperation first"; exit 1
    fi
    require_free_ports 3d
    "$MJPY" local_mock/wrc_mujoco_mirror.py &
    MIRROR_PID=$!
    trap 'kill $MIRROR_PID 2>/dev/null' EXIT
    echo "3D window started (pid $MIRROR_PID). Open http://127.0.0.1:8090/ and type e.g. 'pick and place the red cube'."
    python local_mock/dashboard_mock.py ;;
  mcp-smoke)
    mkdir -p setup_logs
    WRC_RUN_DIR="runs/mcp_smoke_$STAMP" exec python local_mock/mcp_smoke_client.py ;;
  mcp)
    exec python -m wrc_demo.apps.mcp_server ;;
  *)
    sed -n '2,20p' "$0"; exit 2 ;;
esac
