# WRC mock-mode setup report

Written 2026-11-14 on the hackathon laptop. Repo: https://github.com/TheMoonAstronaut/wrc, commit `723121f` (2026-11-11, "feat: update the RS demo and Kimi Code setup"), cloned to
`<project>/wrc`. This is separate from `steel_team/`, `DocsPipline/` and `Obsidian/`, and none of those were touched.

**Scope:** MOCK MODE ONLY. There was no physical hardware, no GPU inference, and no cloud LLM. **Nothing physical moved at any point.** Every "arm motion" below is the kinematic `MockArm` streaming waypoints through the real safety harness.

---

## 1. Environment

| Item | Value |
|---|---|
| OS | macOS 27.2 (Darwin 27.2.0), **arm64** (Apple M1 Pro) |
| RAM / disk free | 16 GB / ~102 GB |
| GPU | Apple M1 Pro (Metal). **No NVIDIA/CUDA.** torch reports `mps=True, cuda=False` |
| Conda | 24.11.3 (`/opt/anaconda3`) |
| Git / system Python | 2.54.0 / 3.14.5 (Homebrew, not used) |
| Docker | `/usr/local/bin/docker` is present but was not needed or used |
| Env Python | 3.10.22 (conda-forge), env `wrc-demo` at `/opt/anaconda3/envs/wrc-demo` |

**Platform support:** the README only documents **Ubuntu 22.04+ / Debian 12+ / WSL2** on **x86_64** ("aarch64 untested"). macOS is not documented. The mock stack is pure Python, so it was installed natively. It works after one build-variant fix (see §3). The real-hardware paths cannot run on this Mac: SocketCAN `can0`, Orbbec/RealSense udev setup, and the vLLM/CUDA local LLM.

Key installed versions: numpy 2.2.6, scipy 1.15.2, torch 2.14.1 (CPU/MPS wheel), ultralytics 8.4.123, pin (Pinocchio) 4.1.0, motorbridge 0.6.0, openai 3.51.0, anthropic 2.8.0, opencv-python-headless (via the package extras), fashionstar-uart-sdk 1.3.12, huggingface-hub 2.4.0, pytest 9.2.0.

## 2. Steps performed

1. `git clone https://github.com/TheMoonAstronaut/wrc wrc` (~477 MB with `models/` and `assets/`).
2. Read `README.md` (Chinese), `CLAUDE.md`, `environment.yml`, `pyproject.toml`, `.env.example`, `configs/**`, `src/wrc_demo/apps/{demo,mcp_server}.py`, `agent/{llm,reflex}.py`, `control/mock_arm.py` and `scripts/setup_agents.py`.
3. `PYTHONNOUSERSITE=1 conda env create -f environment.yml` (log: `setup_logs/conda_env_create.log`). The conda layer succeeded. The pip layer **failed on a transient DNS error** (`files.pythonhosted.org` did not resolve).
4. Re-ran **exactly the same pip block** from `environment.yml` inside the env, with only retry and timeout flags added (log: `setup_logs/pip_install_retry.log`). It succeeded, and `pip check` reports no broken requirements:
   ```bash
   PYTHONNOUSERSITE=1 /opt/anaconda3/envs/wrc-demo/bin/python -m pip install --retries 10 --timeout 60 \
     "pandas>=2.0,<3" "torch>=2.0" -e ".[perception,llm,kinematics,arm,dev]" fashionstar-uart-sdk huggingface-hub
   ```
5. **Fixed a macOS-only crash** (see §3) by switching OpenBLAS to its pthreads build variant (log: `setup_logs/openblas_pthreads.log`).
6. Verified imports: numpy, scipy, cv2, yaml, pandas, torch, ultralytics, pinocchio, motorbridge, openai, anthropic, fashionstar_uart_sdk, huggingface_hub, wrc_demo, `wrc_demo.apps.demo` and `wrc_demo.apps.mcp_server` all import cleanly.
7. Ran the tests (§5), the E2E runs (§6) and the MCP checks (§7).

Not installed, on purpose: `pyorbbecsdk2` (the README makes it an optional post-install step and it is only needed for an Orbbec camera), the CUDA build of torch, and the vLLM / Qwen3-VL server (it needs an NVIDIA GPU). No model weights were downloaded. The mock camera pins a mock detector, so YOLOE is never loaded.

## 3. Config and environment changes

| Change | Where | Why |
|---|---|---|
| `libopenblas 0.3.34` build changed from `openmp_*` to `pthreads_*` (**same version**) | conda env `wrc-demo` only | On macOS, conda-forge's OpenMP OpenBLAS loads the env's `libomp.dylib` and the torch wheel loads its own bundled `libomp.dylib`. The result was `OMP: Error #15 ... libomp.dylib already initialized`, which aborted **every** `import torch` and `import ultralytics`. The pthreads variant removes the second OpenMP runtime. The unsafe `KMP_DUPLICATE_LIB_OK=TRUE` workaround was deliberately **not** used. |
| **No repo files modified.** `configs/demo.yaml` still defaults to `llm.profile: kimi_code` | — | Mock mode is selected on the CLI instead: `--arm mock` (which implies the mock camera and mock detector) plus `--llm mock`. For the MCP server the default is already `WRC_ARM=mock`, and it hardcodes `llm='mock'`. |
| No `.env` created | — | None of the LLM credentials is set on this machine. The missing variables are `KIMI_CODE_API_KEY` (default profile `kimi_code`), `MOONSHOT_API_KEY` (`kimi`), `MINIMAX_API_KEY`, `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`. No local LLM server (Ollama/vLLM) is running either. To use a real LLM: `cp .env.example .env`, fill in **one** key, `set -a; source .env; set +a`, then pass e.g. `--llm anthropic`. That path is **untested**. |
| No `.mcp.json` written and no MCP server registered with any host | — | Registering would change user-level agent configuration. The exact command is in §7. |

**New files** (all inside `wrc/`; `git status` shows only `local_mock/` as untracked, because `runs/` and `setup_logs/` are already gitignored):

- `SETUP_REPORT.md`: this file.
- `START_WRC.sh`: launcher (§9).
- `local_mock/run_mock_task.py`: local driver that runs the **same** `load_demo_config → build_runtime → AgentOrchestrator` pipeline. It adds two opt-in mock-only knobs: `--simulate-contact` (`arm.object_stop_frac = 0.5`, the fixture from `tests/test_orchestrator_e2e.py`) and `--scripted-plan` (a fixed tool-call script standing in for the LLM). Grasp-memory and experience state are written into the run directory.
- `local_mock/mcp_smoke_client.py`: a minimal stdio MCP client used to smoke-test the server.
- `setup_logs/*.log`: raw output of every install, test and E2E run.
- `runs/*`: traces from the E2E runs (trace.jsonl, keyframes, summary.txt).

Side effect outside the repo: `~/.wrc_demo/grasp_memory.json` was created at 20:26 by the test suite and/or the stock CLI runs. That is the repo's default persistent grasp memory, and CLAUDE.md says to delete it for deterministic runs. `local_mock/run_mock_task.py` writes to the run directory instead.

## 4. Repo structure (short)

```
src/wrc_demo/
  apps/        demo.py (CLI + build_runtime composition root), mcp_server.py (stdio MCP),
               stream_server.py (MJPEG dashboard), viewer.py, record.py
  agent/       orchestrator.py (3-tier: reflex regex → experience memory → LLM tool loop),
               reflex.py, llm.py (MockLLM, OpenAI-compat, Anthropic clients), advisor.py, trace.py
  skills/      runtime.py (SkillRuntime.execute + TOOL_SPECS), teach.py
  perception/  camera backends (mock/Orbbec/RealSense/UVC), detector (mock/YOLOE), world model
  grasping/    camera-frame mask planner, GraspGen-X client, analytic OBB planner
  control/     Pinocchio kinematics, MockArm, LazyArm, RebotRSArm (SocketCAN)
  safety/      SafetyHarness + SafeArm (workspace AABB, joint limits, velocity cap, e-stop, watchdog)
  memory/      BeliefStore, EpisodicMemory
configs/       demo.yaml + profiles: cameras/ (mock, d405, orbbec…), arms/ (mock, rebot_rs), llm/ (7 profiles)
tests/         ~420 tests (6 hardware-marked, deselected by default)
scripts/       start_all.sh (real-rig launcher), setup_agents.py (MCP host registration), vLLM/graspgenx servers
```

## 5. Test results

**README quick tests** (`./START_WRC.sh tests`): **17 passed, 1 skipped.** This matches the README's expected result exactly.

**Full suite** (`pytest tests/ -q`, 62 s; log `setup_logs/pytest_full.log`):

| Total collected | Passed | Failed | Skipped | xfailed | Deselected (hardware) |
|---|---|---|---|---|---|
| 420 (414 run) | **381** | **20** | 5 | 8 | 6 |

Hardware-dependent tests: the 6 `@pytest.mark.hardware` tests are deselected by `pyproject` addopts. Of the skips, 3 are hardware or optional-asset skips: no Orbbec in `cameras.json`, and 2 that need `WRC_PHASE6_HARDWARE=1`. One needs the external `rebot_grasp` package, and one needs `mobileclip2_b.ts` in the CWD.

**The 20 failures. None of them was edited or forced to pass.** All of them reproduce after the environment was fully fixed, and none involves a missing package or the platform:

| Group | Tests | Root cause seen |
|---|---|---|
| Test stubs out of date with the code | `test_profiles.py` ×5, `test_profile_switching.py` ×3 | Tests pass plain `dict`s into `make_llm`, which now expects `Cfg` (`'dict' object has no attribute 'type'`), and stub LLMs lack the newer `supports_vision` attribute |
| | `test_persistence_loop.py` ×4 | Stubs `FakeKin` and `StubKin` lack the `n_joints` attribute and the `ik(retries=…)` argument that the code now uses |
| Tool-set changed upstream | `test_teach_skills.py::test_tool_specs_total_6` (now 7 because `wave` was added), `test_mcp_server.py::test_initialize_and_tools_list_without_hardware` (expects `grasp_object`, which is no longer in `TOOL_SPECS`) | Tests not updated after the 2026-09/11 skill changes |
| Python-version mismatch in the tests | `test_setup_agents.py` ×2 | Tests `import tomllib`, which only exists in Python ≥ 3.11, but `environment.yml` pins 3.10. The script itself does not use tomllib |
| Dependency version (likely) | `test_mobileclip_pin.py` ×1 | Expects wrc_demo's ultralytics download patch to be installed. `pyproject` allows any `ultralytics>=8.3`, and 8.4.123 was resolved |
| **Not root-caused** | `test_kinematics.py::test_fk_ik_roundtrip` (IK error 0.39 from the given seed), `test_orchestrator_e2e.py::test_memory_survives_between_tasks`, `test_align_to_rebot_grasp.py::test_bridge_path_empty_grasp_returns_failure_dict` | Could be upstream drift, or a version difference (pin 4.1.0 is a new major version against `pin>=3.0`, ultralytics 8.4). **Not checked against a Linux reference run.** |

CLAUDE.md itself mentions "pre-existing 29 failures" upstream as of 2026-08-14, so a failing tail is expected. Fixing these would mean rewriting upstream code or tests, which is outside a setup task.

Additional checks run directly against the real harness, with the mock arm:
- `tests/test_safety.py` and `tests/test_bridge_safety.py`: all pass.
- Target joint2 = 3.5 rad (limit [0.02, 3.12]): `vet_pose` reports `joint2 out of limits`. Streaming `move_joints` was **aborted mid-trajectory** by `harness.approve()` at q2 = 2.436 rad. It returned `False` and **latched the e-stop**, and the next motion was rejected with `SafetyViolation: e-stopped`.

## 6. End-to-end results (mock)

All run logs are in `setup_logs/` and all traces in `runs/`.

| # | Command | Result |
|---|---|---|
| A | Stock CLI, requested task: `python -m wrc_demo.apps.demo --arm mock --llm mock --no-view --no-serve --task "Survey the workspace, identify the available objects, locate the red block, pick it up, and place it next to the box."` | The task was **received**. The reflex tier did not match the compound sentence, so it went to the LLM tier. The **stock mock LLM is a 2-step wiring check**: it calls `get_observation` (mock camera → mock detector → `red cube` at base (0.29, 0, 0.05)) and then `task_done`. The report says `success: True` but **no pick happened**. That success flag comes from the scripted mock, not from the task. |
| B | Stock CLI REPL: `look around`, `what is on the table`, `pick up the red cube and put it in the box`, `go home`, `open the gripper` | The reflex tier handled `look around` → `get_observation`, `go home` → `move_home` and `open the gripper` → `open_gripper`, all OK. `pick up … put it in the box` was parsed into `pick_and_place(object="red cube", destination="box")` and ran **8 full grasp attempts** (planning → IK → safety vetting → mock motion → gripper verification). Every attempt failed with *"air grasp: gripper closed fully, object not held"*, because the stock CLI's MockArm never simulates contact. The task then fell back to the mock LLM and reported failure. |
| C | `local_mock/run_mock_task.py --simulate-contact --task "pick up the red cube and put it in the box"` | The **grasp succeeded** (`held object: red cube`). The place step failed honestly: `localize 'box' failed: no detections for ['box']`, because the synthetic scene contains only the red cube. |
| D | **`./START_WRC.sh demo`** (`--simulate-contact --scripted-plan`, the requested task) | `success: True`, 7 steps, 2.7 s. The tool sequence was `get_observation ✓ → list_objects ✓ → localize_object(red cube) ✓ (0.29, 0, 0.05) → localize_object(box) ✗ (not in scene) → pick_and_place(red cube) ✓ placed_at (0.18, -0.17, 0.05) = default drop zone → list_objects ✓ → task_done`. The belief store moved the cube to the drop zone, and `trace.jsonl`, `summary.txt` and 13 keyframes were written. |

Pipeline stages, verified vs not:

| Stage | Status |
|---|---|
| Task reception, CLI and REPL | ✅ verified |
| Intelligence layer: reflex tier | ✅ verified (real regex grammar) |
| Intelligence layer: LLM tier | ⚠️ only the **scripted** mock LLM. No real model was run, because no credentials or local server were available |
| Tool/skill selection | ✅ reflex selection is real. The LLM tool calls in D are scripted |
| Camera (mock synthetic frame + depth) and detection (MockDetector, label `red cube`, conf 0.95) | ✅ verified. YOLOE was not exercised |
| Planning: grasp planner + Pinocchio IK | ✅ verified (runs B, C, D) |
| Safety validation: harness vetting, mid-stream abort, e-stop latch | ✅ verified |
| Mock arm execution | ✅ verified (kinematic simulation only) |
| Completion report, trace.jsonl, keyframes, belief/episodic memory | ✅ verified |
| Limitation | The mock camera image is static: the cube is always drawn at the same pixel location. Only the belief store reflects the move. |

## 7. MCP status

- **Implementation:** `src/wrc_demo/apps/mcp_server.py` is a hand-written MCP server using newline-delimited JSON-RPC 2.0 over **stdio**. It supports `initialize`, `tools/list`, `tools/call` and `ping`, with protocol versions 2025-06-18, 2025-03-26 and 2024-11-05. It is launched with `python -m wrc_demo.apps.mcp_server` or the `wrc-mcp` entry point. Configuration comes from environment variables: `WRC_ARM` (default `mock`), `WRC_CAMERA(S)`, `WRC_RUN_DIR`, `WRC_VIEW`, `WRC_STREAM`, `WRC_STREAM_PORT`, `WRC_PREWARM`, `WRC_DETECTOR_MODEL` and `WRC_DETECT_CLASSES`. The MCP host acts as the brain, and the built-in LLM is hardcoded to `mock`. The arm is a LazyArm and is only powered on the first motion call.
- **Verified** (`./START_WRC.sh mcp-smoke`, log `setup_logs/mcp_smoke.log`): `initialize` returned `wrc-demo 0.1.0` with protocol 2025-06-18, and **`tools/list` returned 11 tools**:
  `list_objects, pick_and_place, move_home, wave, teach_record, teach_replay, camera_snapshot, world_state, live_view_url, emergency_stop, reset_stop`.
  The README and CLAUDE.md say 10; `wave` was added since.
  Calls that worked: `world_state` (red cube visible), `list_objects`, `camera_snapshot` (returned a JPEG image), `move_home` and `ping`. `pick_and_place(red cube)` ran and returned `isError=True` with *air grasp*. That is expected, because MCP mode has no contact simulation. The server exited cleanly with code 0.
- `teach_record` and `teach_replay` were **not called**: they reach for the PiPER Mate serial device and the real arm.
- **Connecting an external agent (not done).** The first command is the repo's generator in print-only mode. The `claude mcp add` line is adapted from its output, with `WRC_VIEW=0`, `WRC_STREAM=0` and `PYTHONNOUSERSITE=1` added and `PYTHONPATH` dropped, since the package is installed editable:
  ```bash
  /opt/anaconda3/envs/wrc-demo/bin/python scripts/setup_agents.py --host claude --arm mock --camera mock \
      --python /opt/anaconda3/envs/wrc-demo/bin/python            # add --write to create .mcp.json
  # or user-scoped:
  claude mcp add --scope user --env WRC_CAMERAS=mock --env WRC_ARM=mock --env WRC_VIEW=0 --env WRC_STREAM=0 \
      --env PYTHONNOUSERSITE=1 wrc-demo -- /opt/anaconda3/envs/wrc-demo/bin/python -m wrc_demo.apps.mcp_server
  ```
  The generator sets `DISPLAY=:1`, which is meaningless on macOS. Use `WRC_VIEW=0` instead. Registering with a real MCP host such as Claude Code, Codex or Hermes is **untested**.

## 7b. Live dashboard and 3D view (added after the first report)

**WRC dashboard** (`./START_WRC.sh dashboard` → http://127.0.0.1:8090/): this is WRC's own stdlib web UI, with the camera tile, a chat box, a STOP button, live robot narration and the world-model table. It is wired by `local_mock/dashboard_mock.py`, which uses the same `build_runtime` + `AgentOrchestrator` as `scripts/dashboard_runner.py`, with these differences: a mock LLM, simulated grasp contact, binding to **127.0.0.1 only**, and no 60 s wait for a "pink object". Verified: `look around` → ok, and `pick and place the red cube` → `pick_and_place → ok` (reflex tier, ~2.4 s).

**3D view** (`./START_WRC.sh 3d`). This is WRC driving a MuJoCo rendering of the reBot B601-RS:
- `local_mock/mirror_publisher.py`, a thread in the WRC process, streams the mock arm's joint angles, the gripper value, `held_object`, the beliefs and the drop zone as JSON over UDP to `127.0.0.1:9870` at 50 Hz. It only reads state and cannot command motion.
- `local_mock/wrc_mujoco_mirror.py` runs in the **teleoperation venv**, `steel_team/teleoperation/.venv`, because that env has mujoco 4.0.1. It loads Seeed's official `RS/mjcf/rs_arm.xml` from `steel_team/teleoperation/assets/`, which is fetched by that folder's `setup_assets.sh` and is gitignored. It copies WRC's joints onto the model 1:1: kinematic playback with no physics and no IK.
- **Mapping verified:** WRC's RS URDF (`safety.harness.Kinematics`, `gripper_end`) and the RS MJCF (`tcp` site) give the same TCP pose to ~4×10⁻⁸ m for 10 joint vectors (home, zero, 8 random), and both share the same base frame. Gripper: `joint7 = 0.05 × (1 − WRC grip)`, where 0 = closed and 0.05 = open.
- The red cube is carried while the jaws are closed on it, with grip between 0.1 and 0.95. A full-rate log showed WRC's grasp at TCP z = 0.030 m with the jaws at 23 % closed (a 69 mm opening for the 7 cm cube). WRC only sets `holding` after lifting and verifying the grip, about 1 s later. Translucent boxes show WRC's beliefs: red = visible, grey = remembered.
- Verified: headless snapshots of grasp → lift → carry → place, plus the live window log (`jaws closed on the cube at (0.29, 0.00, 0.030)` → `released at (0.18, -0.17)`). Both were driven through the dashboard chat and through `./START_WRC.sh 3d`.
- Caveats:
  - The mock camera is still a fixed image, so WRC keeps "seeing" a cube at (0.29, 0). A second pick therefore grabs again at the start position, and the 3D view moves its cube into the jaws there.
  - Perception is not rendered from MuJoCo.
  - On macOS the window needs `mjpython`. For the uv venv I added a symlink, `steel_team/teleoperation/.venv/lib/libpython3.13.dylib`, pointing to uv's Python library (inside the gitignored `.venv`).

## 7c. reBot B601-DM port with Claude as the planner (added last)

Full details, the verified runs and the real-arm bring-up checklist are in **`local_dm/README.md`**. Summary:

- **What it is:** WRC running on the team's actual arm model, the B601-**DM**, with Claude deciding the steps from the camera image. Start with `./START_WRC.sh dm`.
- **Verified in a physics simulation** of the DM arm (MuJoCo, servo-driven joints, contact grasping, rendered RGB-D camera, WRC's real YOLOE detector): five tasks succeeded, checked against the simulator's ground truth to 5–15 mm, and one out-of-reach task failed honestly.
- **Not verified:** the real arm (`--robot real` has never run on hardware), a real camera and its calibration, a fresh install on another machine, Linux.
- **No upstream WRC file was modified.** Added: `local_dm/` (bridge server, WRC arm and camera backends, Claude client, launcher, installer, README), `configs/arms/rebot_dm.yaml`, `configs/cameras/dm_sim_overhead.yaml`, and `dm` modes in `START_WRC.sh`.
- **Environment changes made for this:**
  - `uv` was installed with Homebrew, and `steel_team/teleoperation` was set up with its own documented commands (assets and `.venv`, both git-ignored there). A `libpython3.13.dylib` symlink was added inside that `.venv` so `mjpython` can open a window on macOS.
  - On first use of YOLOE's text prompts, ultralytics downloaded `models/mobileclip2_b.ts` (254 MB) and pip-installed `clip`, `ftfy`, `regex` and `wcwidth` into the `wrc-demo` env by itself.
  - A symlink `mobileclip2_b.ts` → `models/mobileclip2_b.ts` sits in the repo root, as the WRC README suggests.
  - `wrc/.env` holds the Anthropic API key (git-ignored, mode 600). It is not in the share zip.
- **Camera calibration helper** (`local_dm/calibrate_camera.py`, printed marker sheet): verified in the sim against the true camera pose (table points within 4.3 mm; pick-and-place with the calibrated camera worked with depth and RGB-only). Untested on a real camera. `./START_WRC.sh dm-calib-test` repeats the check.
- **Share package:** `../wrc-dm-port.zip` (the added files only; install steps are in `local_dm/README.md`).

## 8. Limitations and blockers

- **Real hardware (reBot B601-RS over SocketCAN, Orbbec/RealSense cameras, PiPER Mate teach): blocked on this Mac.** It needs Linux (SocketCAN, udev), the devices, and calibration. Nothing was connected or attempted.
- **Real LLM reasoning: blocked by missing credentials.** None of `KIMI_CODE_API_KEY`, `MOONSHOT_API_KEY`, `MINIMAX_API_KEY`, `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set. The local `local_qwen3_vl` profile needs an NVIDIA GPU and a vLLM server. The free path would be an OpenAI-compatible local server (e.g. Ollama) through `configs/llm/openai.yaml` with `OPENAI_BASE_URL`, but that is not installed and **untested**.
- **YOLOE open-vocabulary detection: untested.** Mock mode uses MockDetector. Weights for it exist under `models/`.
- The MJPEG dashboard (`stream_server`, port 8090) and the OpenCV viewer were not exercised. All runs used `--no-serve --no-view`.
- macOS is not an officially supported platform. The OpenBLAS variant fix (§3) is required, and it must be repeated if the env is recreated with `conda env create`.
- 20 failing unit tests from upstream test/code drift (§5). Three of them are not root-caused.

## 9. Restart commands

```bash
cd <project>/wrc
./START_WRC.sh                  # interactive mock REPL (stock CLI)
./START_WRC.sh demo             # verified E2E: survey → locate → pick → place (scripted LLM, simulated contact)
./START_WRC.sh task "go home"   # single task through the stock CLI
./START_WRC.sh dm               # B601-DM physics sim + dashboard, Claude plans (see local_dm/README.md)
./START_WRC.sh dm-task "put the red block on the blue block"   # one DM task, headless
./START_WRC.sh 3d               # dashboard + live MuJoCo 3D window mirroring WRC's arm (Ctrl+C stops both)
./START_WRC.sh dashboard        # WRC live web dashboard → http://127.0.0.1:8090/ (verified: "pick and place the red cube" → ok)
./START_WRC.sh tests            # README quick tests (17 passed, 1 skipped)
./START_WRC.sh tests-full       # full mock suite (381 passed / 20 failed / 5 skipped last run)
./START_WRC.sh mcp-smoke        # MCP server + tools/list + sample calls
./START_WRC.sh mcp              # raw MCP stdio server for an MCP host
```

Manual equivalent:

```bash
conda activate wrc-demo
export PYTHONNOUSERSITE=1 YOLO_OFFLINE=True
python -m wrc_demo.apps.demo --arm mock --llm mock --no-view --no-serve --interactive
```

To recreate the env from scratch:

```bash
PYTHONNOUSERSITE=1 conda env create -f environment.yml          # re-run the pip block (§2 step 4) if pip hits a network error
conda install -n wrc-demo -c conda-forge -y "libopenblas=*=*pthreads*"   # macOS OpenMP fix
```

## 7d. Team real-workspace simulation (2026-11-15)

`./START_WRC.sh dm` now runs the team's `rebot_sim` scene (steel_team main, `simulation/`) when it is found; `WRC_DM_SCENE=demo` keeps the old table. On this machine the team's main branch is checked out as a separate worktree in `../steel_team_main` (the shared `../steel_team` checkout was not touched). What changed, what was measured and what is still open: `local_dm/README.md`, section "Real-workspace simulation". Simulation only; no hardware was connected.

## 7e. Measuring objects without a depth camera (2026-11-15, fourth push)

On the real robot the webcams give no depth, so standing objects were placed and sized wrongly and grasps missed. `local_dm/two_view.py` now measures each object from its outline in two calibrated scene cameras (one camera as a fallback, optional table of known sizes). Details, usage and limits: `local_dm/README.md`, first section.

- Tested: `./START_WRC.sh dm-measure-test` (maths on drawn scenes; wiring with WRC's data types and a stand-in robot). Both pass.
- Simulator with depth switched off and the team's two cameras: measured positions within 3 mm of the truth (18 measurements); 8 of 8 random pick-and-place trials into the bin succeeded (keyword commands). Control with the old flat-table method: 0 of 4.
- Not run: Claude planning with depth off, the real cameras, the real arm.
