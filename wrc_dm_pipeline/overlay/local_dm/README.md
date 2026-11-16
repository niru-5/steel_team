# WRC on the reBot Arm B601-DM

Prompt → Claude looks at the camera → picks a WRC skill → WRC plans the grasp, checks safety and moves the arm.

This folder ports the WRC pipeline (built for the B601-**RS**) to the B601-**DM**. Written 2026-11-14 at the Leukerbad hackathon. No upstream WRC file is modified; everything here is additive.

## Latest version (fourth push, 2026-11-15): measuring objects without a depth camera (works in the simulator, untested on the real robot)

**The fault on the real robot.** The webcams give no distance. WRC then assumes every pixel lies on the table. That is right for the table and wrong for anything standing on it: in the wiring test a 13 cm can looks 0 cm tall and its middle is placed 8 cm away from where it stands. The arm closes at the wrong height and in the wrong place.

**The fix.** With two fixed, calibrated scene cameras, each object is measured from its outline in both pictures (`two_view.py`):

- Seen from above, an outline spans a range of directions from the camera. The object's middle lies on the line through the middle of that range. Two cameras give two lines, and where they cross is the object's position.
- That range of directions times the distance is the width.
- The highest outline point is the object's top edge. Its line of sight, followed out to the object's distance, gives the height. A camera near table level is best for this.
- The gripper then closes at half the measured height (50 % straight down, 55 % leaning), as before.
- If the second camera does not see the object, one camera alone is used: the lowest outline point is where the object touches the table. This is less exact.
- Optional: a table of known sizes (`--known-objects`, see `known_objects_example.yaml`). When the detector's name and the measured size both fit an entry, the exact size from the table is used. The cameras still decide where the object stands.

The gripper camera is not used for this.

**How to use it on the real setup**

1. Calibrate both scene cameras with the printed sheet (`./START_WRC.sh dm-calibrate`, section "Camera calibration" below). Leave the sheet in the same place for both. Give each camera its own profile name and its own `device_index`.
2. Check what the cameras measure. This moves nothing and needs no arm:
   ```bash
   ./START_WRC.sh dm-measure --camera my_cam --camera2 my_cam2 --table-z 0.0 --classes "cup,red cup,white box"
   ```
   It prints each object's position, height and width, what each camera alone says, and saves pictures with the outlines and the measured foot and top drawn in. Put the can at a spot measured with a ruler and compare.
3. Run the pipeline with both cameras:
   ```bash
   python local_dm/run_dm.py --camera my_cam --camera2 my_cam2 --table-z 0.0 --classes "cup,red cup,white box" --known-objects my_objects.yaml --task "..."
   ```
   Measuring from outlines switches on by itself when the main camera has no depth (`--measure auto`).

**What was tested, and what was not**

| What | Result |
|---|---|
| The maths, on drawn scenes with exact outlines (`./START_WRC.sh dm-measure-test`): can, cup and box at 7 spots, 3 camera arrangements | Two cameras: position within 4.2 mm, height within 5.7 mm (can and cup: 1.2 mm), can and cup width within 1.6 mm. One camera: can within 2.5 mm; cup position 16–17 mm off and height up to 20 mm off (its foot is narrower than its rim) |
| Outlines one pixel too big or too small | Position within 6 mm, height within 6 mm; can and cup width up to 5 mm off |
| Calibration wrong by 0.5° and 5 mm per camera | Position up to 26 mm off (about 12 mm on average with both cameras raised, 4–5 mm with one camera near table level). **The calibration is what limits the accuracy** |
| Pairing with two objects in view and swapped names; known-size table; second camera blocked | Pass |
| Wiring into the pipeline (WRC's real data types, stand-in robot): planned grasp vs. true object | Can 0.1 mm, cup 0.5 mm from the true middle, at half height; the 9 cm cup gets the leaning grasp first |
| Simulator, depth switched off, real detector, the team's two cameras (`dm-measure --rgb-only`): can and cup at 9 random layouts, compared with the simulator's true positions | Position within 3 mm in all 18 measurements. Height within 6 mm (can 12.6–13.2 cm for 13.0, cup 11.1–11.6 for 11.0). Width 2–5 mm too wide, once 10 mm (can). One measurement fell back to one camera and was still 2 mm off |
| Simulator, depth switched off, full pick-and-place into the bin, 8 random trials, keyword commands (`dm-trials --no-llm --extra "--rgb-only --camera2 dm_sim_overhead"`) | **8 of 8** (4 can, 4 cup, both bin compartments), about 31 s each |
| Control: the same trials with the old flat-table method (`--rgb-only --measure depth`, 4 trials) | **0 of 4.** Cup: "required width 115 mm > gripper max 91 mm". Can: "air grasp: gripper closed fully, object not held". This is the fault seen on the real robot |
| Simulator with Claude planning and depth off | **NOT RUN** (the trials used keyword commands; the planner is not what changed) |
| Real cameras, real arm | **NOT RUN** |

**Limits**

- **Lens data.** The geometry needs each camera's true focal length. A profile without `intrinsics` guesses a 70° field of view, and lens distortion is not modelled. A wrong focal length shows up in `dm-measure` as a wrong height for an object of known height.
- A camera that looks at an object from almost straight above (more than 68° down) cannot see its height and is not used for that object.
- An object partly outside ONE camera's picture is still measured, using the side edge that is visible and the width from the other camera (found in the simulator: a can at the table edge was 16 mm off before this and 0 mm after). If no usable combination is left, the pick stops with "cannot measure ...: cut off at the edge of the camera picture" instead of reaching for a wrong place.
- An object partly hidden by the arm or by another object gives a wrong size, and nothing detects that. Objects are measured with the arm at home.
- The simulator's pictures are clean: no lens distortion, exact calibration, even light. Real pictures will be worse.
- Objects under 4 cm tall keep WRC's flat-table estimate, which is right for flat things.
- Objects wider at the bottom than at the top (a bottle) can read up to about 2 cm too tall from a steep camera; the camera near table level corrects most of it.
- The live object list on the dashboard still uses the flat-table guess. Only the object being picked is measured, at the moment it is located.
- For a real arm the bin, the walls and the table height still have to be given by hand (`--table-z`, `--classes`); bin placement needs the workspace settings that only the simulator provides today.

## Third push (2026-11-15): the team's real-workspace simulation

`./START_WRC.sh dm` now runs the team's own MuJoCo replica of the real workspace (`steel_team` main, `simulation/`, package `rebot_sim`): the measured table, the red can, the paper cup, the foam bin with two compartments, and the corner camera `cam_corner`. It is used automatically when the team checkout has `simulation/` (it is on `main`) and `uv sync` was run in `teleoperation/`. `WRC_DM_SCENE=demo ./START_WRC.sh dm` brings back the old three-block table.

What had to change for this scene (all in `local_dm/`, nothing in WRC or in the team's code):

- **Side-view camera.** WRC's first-choice grasp dives along the camera's line of sight. From a corner camera that closes on air. Standing objects are now located by the middle of their top (1–2 mm from the true centre in this scene) and grasped from above.
- **The cup is 9 cm wide, the gripper opens 9.1 cm.** Straight down, a finger ends up inside the cup. Wide objects, and objects too far for a straight-down pose, are grasped with the gripper leaning in from the robot's side.
- **The bin.** `bin_0` (nearer the robot) and `bin_1` are named places: "put the can in the bin". The arm cannot reach over the bin pointing straight down, so it leans outwards as little as needed and lets go above the rim.
- **Tall objects** are put down at the height and angle they were gripped at, instead of WRC's fixed 5 cm.
- Detector words for this scene: `cup, red cup, white box`. YOLOE calls the plain red can "red cup" and the bin "white box"; Claude is told so.
- `dm-shuffle` and `dm-trials` use the team's own randomizer (`spawn_region`) in this scene.

Measured (simulation only, judged on the simulator's ground truth "is the object inside the named compartment"):

- Default layout, Claude planning, free-form sentences: can → `bin_0` and cup → `bin_1`, both succeeded (about 30 s each, 3 Claude calls).
- 8 randomized keyword trials **before** the leaning grasp: 5 of 8. The 3 failures (cup hooked on a finger; cup and can too far for a straight-down grasp) were each re-run after the fix and succeeded. **A full randomized re-run after the fix has not been done, so there is no new success rate.**

Limits to know before the real arm:

- **Depth.** WRC needs a depth image. The simulator renders one from `cam_corner`; the real corner camera (a webcam) has none. On the real setup use a depth camera, or WRC's flat-table depth for RGB-only cameras, which is wrong for tall objects such as the can and the cup. Not solved here.
- The detector sometimes calls the can "cup" when it stands far from the camera. Claude can still pick by colour; the keyword commands cannot.
- Objects dropped into the bin may fall over inside it. The can and cup are light; nothing checks for breakable contents.
- The team's random start places reach x = 0.52 m. Not every one of them was tested.

## Second push (2026-11-15): fixes from simulator testing

Changes since the first push, all found by testing in the physics simulation (old three-block demo table):

- **Joint ranges.** WRC planned with Seeed's URDF ranges (wrist roll ±180°). The real arm, and the simulator that mirrors it, accept ±90°. Commands were clipped silently and the gripper arrived in the right place turned about 50° wrong. WRC's arm model is now clipped to the real ranges, every grasp is also offered turned half a turn, and the arm driver refuses to start if the two disagree.
- **A pose the arm did not reach is now a failure**, not a silent success.
- **Objects are put down with the gripper straight down.** WRC's default tilts the tool 21°, which left tall objects leaning 30°.
- **"Pick X" with no destination puts the object on a free spot** chosen before each task and checked for reach, so objects no longer pile up on one place. (`--drop-zone x,y` fixes the spot.)
- **Nothing is left in the gripper.** If a task ends while still holding an object, the robot sets it down on a free spot.
- **Claude and detector names.** Claude is told that the detector's names can differ from the user's words (a cylinder seen from above is a "ball") and that "remembered" entries are old sightings, not extra objects.
- **Every Claude decision is shown** in the dashboard and the terminal with its API request ID.
- **The 3D window is display only.** Closing it no longer stops the simulator, and a key pressed in it that resets MuJoCo's scene is undone. Type tasks in the web page.
- **Real-arm path** now follows the team's `teleoperation/go_ready.py` (measured gripper range, lag guard). Still untested on hardware.
- New commands: `./START_WRC.sh dm-shuffle` (move the sim objects), `dm-trials` (randomized trials with a success rate), `dm-calibrate` (camera calibration).

Measured before these fixes: 15 of 20 randomized Claude trials succeeded. The 5 failures were the joint-range bug (3) and the name mismatch (2); each failing case was re-run after the fix and now succeeds. The full 20-trial re-run after the fixes has not been completed, so there is no new success rate yet.

**The reach figures further down were measured before the joint-range fix and the straight-down put-down. Treat them as approximate.**

## What is verified and what is not

| Part | Status |
|---|---|
| DM kinematics inside WRC (Seeed's DM URDF + a `gripper_end` tool frame) | **Verified.** Matches Seeed's DM MuJoCo model to 1e-10 m on 10 poses |
| Full pipeline on a **physics simulation** of the DM arm: camera → YOLOE detection → Claude → grasp planning → IK → safety harness → servo motion → contact grasp → place | **Verified.** See "Verified runs" below |
| Claude as the planner (`claude-opus-5-5` by default) | **Verified** with a real API key, 2–4 s per step |
| Web dashboard + 3D physics window, one command | **Verified** on macOS (Apple silicon) |
| **Real B601-DM arm** (`--robot real`) | **UNTESTED on hardware.** No arm was connected. It is wired the way the team's `teleoperation/go_ready.py` drives the arm (pos_vel control, the measured gripper range), and was rehearsed with a stand-in driver (`fake_real_arm.py`): joint units, gripper range, lag guard and clean shutdown behave as intended |
| Camera-to-base calibration helper (`calibrate_camera.py`, printed marker sheet) | **Verified in the sim** against the true camera pose: table points within 4.3 mm. A pick-and-place with the calibrated camera succeeded, with depth and RGB-only. See "Camera calibration" |
| Calibration on a **real camera** | **Untested.** Real lenses distort, and the result depends on two ruler measurements |
| Fresh install on another machine with `install.sh` | **Untested** on a clean machine. It ran here, where everything was already installed |
| Linux | **Untested.** Developed on macOS |

Nothing here has moved a physical robot.

## How it fits together

```
 you type a task (dashboard at http://127.0.0.1:8090/ or --task)
        │
        ▼
 WRC AgentOrchestrator ── Claude (claude_llm.py): sees the camera image + object list, calls ONE tool per step
        │                 tools: list_objects, pick_and_place(object, destination?), move_home, task_done
        ▼
 WRC SkillRuntime: detection (YOLOE) → 3D position from depth → grasp plan → IK on the DM URDF
        │
        ▼
 DmBridgeArm (wrc_dm_plugin.py): every 50 Hz waypoint passes WRC's SafetyHarness
        │   (workspace box, table clearance, joint limits, velocity cap, e-stop), then goes out over TCP
        ▼
 dm_bridge_server.py  (runs in the teleoperation venv)
        ├── --robot sim   MuJoCo physics of Seeed's B601-DM model + overhead RGB-D camera   ← verified
        └── --robot real  LeRobot rebot_b601_follower, wired like teleoperation/go_ready.py  ← UNTESTED on hardware
```

Two Python environments are involved, and they only talk over a localhost socket:
- **`wrc-demo`** (conda, Python 3.10): WRC, the detector, Claude client.
- **`steel_team/teleoperation/.venv`** (uv, Python 3.12+): MuJoCo, and LeRobot for the real arm.

## Installing on another machine

This port is a set of files that go on top of a WRC clone. It contains no WRC code, no models and no API key. From the team repo (branch `wrc-dm-pipeline`):

```bash
bash wrc_dm_pipeline/setup_wrc.sh        # clones WRC next to the team repo (commit 723121f) and copies these files in
cd ../wrc
PYTHONNOUSERSITE=1 conda env create -f environment.yml
bash local_dm/install.sh                 # tells you what is still missing
```

`teleoperation` in the team repo must be set up too (`./scripts/setup_assets.sh && uv sync`). If the team repo folder is not called `steel_team`, set `STEEL_TEAM_TELEOP_DIR=/path/to/teleoperation`. `SETUP_REPORT.md` has the details of the WRC environment, including the macOS fix.

## Quick start (simulation)

Prerequisites: the WRC repo with its `wrc-demo` conda env (see `SETUP_REPORT.md`), and `steel_team/teleoperation` set up (`./scripts/setup_assets.sh && uv sync`). If `steel_team` is not next to the `wrc` folder, set `STEEL_TEAM_TELEOP_DIR=/path/to/steel_team/teleoperation`.

```bash
cd wrc
bash local_dm/install.sh                     # one-time: checks both envs, downloads the detector's text encoder
echo 'ANTHROPIC_API_KEY=...' > .env && chmod 600 .env    # your own key; .env is git-ignored
./START_WRC.sh dm                            # physics window + dashboard
```

Open http://127.0.0.1:8090/ and type, for example:

- `Take the red block and stack it on top of the blue one.`
- `Put the yellow thing on the blue block`
- `What objects can you see on the table right now?`
- `Move the red block to the free spot`

One task without the dashboard: `./START_WRC.sh dm-task "put the red block on the blue block"`.

Without an API key the pipeline still runs, but only WRC's built-in keyword commands work: `look around`, `go home`, `open the gripper`, `pick and place the red block`, `put the red block on the blue block`.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` (env or `wrc/.env`) | – | Enables Claude. Never commit it |
| `WRC_CLAUDE_MODEL` | `claude-opus-5-5` | e.g. `claude-fable-5-2` (most capable, about 2× the price, slower steps), `claude-sonnet-5-5`, `claude-haiku-5-5` |
| `WRC_CLAUDE_EFFORT` | `medium` | `low` is faster; `high` thinks longer |
| `WRC_CLAUDE_FALLBACKS` | `1` | Server-side refusal fallback on models that support it; `0` turns it off |
| `run_dm.py --classes "red block,blue block,yellow ball"` | as shown | The detector's vocabulary. **Objects are found only if a label here matches them.** Change it for your objects |
| `run_dm.py --reflex` | off with Claude | Try WRC's keyword shortcuts before Claude. Faster and free, but a free-form sentence can be misread |
| `run_dm.py --table-z 0.03` | 0.03 | Table height above the arm's base plane, in metres. The sim table is 3 cm high |
| `run_dm.py --set key=json` | – | Any WRC config value, e.g. `--set safety.max_joint_vel=0.3` |

Arm profile: `configs/arms/rebot_dm.yaml`. Sim camera profile: `configs/cameras/dm_sim_overhead.yaml`.

## Where the arm can grasp

With a camera looking straight down, WRC grasps straight down. Measured with WRC's own IK on the DM model, from the home pose, table at 3 cm:

| Object | Reachable area (x = distance in front of the arm base) |
|---|---|
| Low object (grasp ~3.5 cm above the table) | x from 0.16 m to **0.40 m**, y within ±0.16 m (to ±0.24 m up to x = 0.36 m) |
| Tall object (grasp ~7 cm above the table) | x from 0.16 m to **0.34 m**, y within ±0.08 m (to 0.32 m at ±0.16 m) |
| Placing (WRC tilts the tool 21° for placing) | up to x = 0.48 m |

Objects outside this area fail with `pregrasp IK failed`. WRC retries 8 times, then Claude reports the failure and parks the arm.

Keep objects out from under the arm at home: the arm hides roughly x < 0.27 m near y = ±0.12 m and x < 0.22 m along y = 0 from an overhead camera.

## Verified runs (simulation, 2026-11-14)

Positions are the simulator's ground truth, read only for checking. WRC itself used the camera.

| Task | Planner | Result |
|---|---|---|
| `pick and place the red block` | keyword shortcut, no LLM | Cube moved 28 cm to the drop zone, 6 mm from the target. 23 s |
| `Take the red block and stack it on top of the blue one.` | Claude, 3 calls | Cube resting on the blue block, 5 mm off-centre. 28 s |
| `Put the yellow thing on the blue block` | Claude | Cylinder standing on the blue block, 1.5 cm off-centre. 48 s |
| `What objects can you see on the table right now?` | Claude | Correct description, including that the blue block was hidden under the cylinder. 3 s |
| `Move the red block to the free spot` | Claude | Cube at the drop zone, 7 mm from the target. 28 s |
| `Put the yellow thing on the blue block` with the cylinder at x = 0.46 m | Claude | **Failed honestly:** out of reach, 8 attempts, arm parked, failure reported |

One failure worth knowing: with the keyword shortcuts on, "stack it on top of the blue one" was read as destination "top of the blue one". The grasp succeeded, the place failed, and the cube stayed in the gripper. That is why Claude reads every instruction by default.

## Camera calibration (printed marker sheet)

WRC must know where the camera is relative to the robot base. WRC's own calibration tools drive the RS arm's SDK, so use this instead. It needs no arm motion: a printed sheet, a ruler and one camera frame.

1. **Print** `local_dm/calib_sheet_A4.png` on A4 at 100 %. Lay it flat on the table, printed side up, anywhere in the camera's view and at any rotation. Keep the arm and objects off it.
2. **Measure** where the two crosses, A and B, are in the robot base frame, in metres:
   - origin: the joint-1 rotation axis, at the base mounting plane
   - +x: straight ahead of the arm, +y: to the arm's left, +z: up

   Create the input file and fill in A, B and `table_z` (the table surface height above the base plane):
   ```bash
   ./START_WRC.sh dm-calibrate --example-config > my_calib.yaml
   ```
3. **Run** it against your camera profile (`uvc`, `d435i`, `d405`, `orbbec_overhead`, …):
   ```bash
   ./START_WRC.sh dm-calibrate --camera uvc --config my_calib.yaml --write-profile my_camera
   ```
   `START_WRC.sh` activates the `wrc-demo` environment for you and works from any folder; file paths are relative to where you run it. Calling `python local_dm/calibrate_camera.py` directly only works inside `wrc/` with the environment active. `./START_WRC.sh dm-calibrate` with no arguments prints these steps.
   This writes `configs/cameras/my_camera.yaml`: a copy of the profile with the measured `extrinsics.T` and `table_plane_cam`. It also saves a debug image with red dots where the fit puts the base origin and the +x / +y directions. **Look at it**: the dots must sit on the real arm base and point the right way.
4. **Use** it: `python local_dm/run_dm.py --camera my_camera --table-z <table_z> ...`

Redo the calibration whenever the camera, the table or the robot base moves.

What to expect, measured in the sim where the true camera pose is known (`./START_WRC.sh dm-calib-test` repeats this):

| Case | Error of points on the table |
|---|---|
| Crosses measured exactly, sheet rotated 35° and off-centre | mean 1.8 mm, max 4.3 mm across the image |
| Each cross mis-measured by 5 mm | 5–16 mm. Worst at the image edges, far from the sheet |
| Pick-and-place with the calibrated camera, with depth | cube stacked 7 mm off-centre |
| Same, RGB only (no depth) | cube stacked 7 mm off-centre; a 7 cm tall object was located 11 mm off |

Notes:
- **Measure the crosses carefully.** That measurement is the accuracy limit, and its error grows with distance from the sheet. Extra markers spread over the table (the `markers:` block) reduce this.
- **RGB cameras without depth** work for objects resting on the table: WRC casts each pixel onto the table plane. The taller the object and the further from the point under the camera, the larger the error.
- **Approximate intrinsics are tolerable for the table plane.** With a `uvc` profile WRC guesses the lens from `fov_deg`. The fit absorbs that for points on the table, not for tall objects. Lens distortion is not modelled, so prefer the centre of a wide-angle image.
- To rehearse an RGB-only camera in the sim, add `rgb_only: true` to a copy of the sim camera profile.

## Real arm bring-up (UNTESTED on hardware — read all of it first)

The bridge's real-arm path has never driven the physical arm. It uses the same calls as the team's own `teleoperation/go_ready.py`, which the team has run on the arm: `pos_vel` control, the measured gripper range, commands streamed at 50 Hz. Treat the first session as bring-up, with one person at the power switch.

You can rehearse this path without an arm: `<teleop>/.venv/bin/python local_dm/fake_real_arm.py --headless` starts the bridge in real-arm mode against a stand-in driver. It checks the bridge's logic, not the arm.

1. **Prerequisites, from the team's teleoperation folder:** `uv sync --extra lerobot`, and the gripper range measured with `tools/measure_gripper.py` (`config/gripper_limits.json`). The bridge refuses to start without it, like the team's scripts.
2. **Bring the arm to a known pose** with the team's script: `python go_ready.py --port <port>`. Stop it, then start the bridge. It begins by holding the pose the arm is in; it does not move by itself:
   ```bash
   <teleop>/.venv/bin/mjpython local_dm/dm_bridge_server.py --robot real --confirm-real --serial-port <port>
   ```
   The 3D window mirrors the measured pose. Compare it with the real arm. If a joint is mirrored or offset, stop.
3. **First motion, slow.** The bridge passes the team's limits to the driver (6° per command, 60°/s). Also cap WRC's own speed for the first runs:
   ```bash
   python local_dm/run_dm.py --task "go home" --llm mock --camera mock --table-z 0.0 --set safety.max_joint_vel=0.3
   ```
   `go home` moves to `home_q` in `configs/arms/rebot_dm.yaml` (arm pulled back and raised). Check that pose is clear of your camera mount and table before you run it. The team's ready pose is a different pose.
4. **Gripper.** Commands are mapped linearly between the measured open and closed angles, staying 5° inside both ends. The bridge assumes the jaw gap is linear in motor angle and 9.1 cm when fully open; if grasp detection misbehaves, check that assumption (`MAX_GAP_M` in `dm_bridge_server.py`). Grip force is whatever the driver's position control gives: watch the first grasps.
5. **Camera.** Calibrate it with the marker sheet (section "Camera calibration" above) and pass the generated profile with `--camera`. Camera types in WRC: `uvc` (RGB only), `d435i`, `d405`, `orbbec_overhead`. The helper is verified in the sim only; on the real camera, check the debug image and then check `look around` reports sensible positions before any pick.
6. **Table height and workspace.** Set `--table-z` to your table's height above the arm base plane, and check `safety.workspace` in `configs/demo.yaml` against your table.
7. **Detector vocabulary.** Set `--classes` to names that match your real objects, and check `look around` lists them before any pick.
8. **Stopping.** The dashboard STOP button and Ctrl+C make the arm hold position. They do not cut power. When the bridge exits, commands stop and the arm may go limp, so **support the arm before closing the bridge**.
9. **Lag guard.** If the arm stays more than 20° behind its command for 1.5 s (blocked, or a wrong mapping), the bridge holds and refuses further motion until WRC resumes. Find the cause before resuming.

Known gaps for the real arm: no hardware e-stop integration, plain position control without the team's newer MIT feed-forward loop, grasp detection relies on the measured gripper opening, and the team's hardware plan has an RGB table camera without depth, which limits WRC to objects resting on the table.

## Files

| File | Runs in | Purpose |
|---|---|---|
| `dm_bridge_server.py` | teleoperation venv | DM arm server: MuJoCo physics (`--robot sim`) or the real arm (`--robot real`, untested) |
| `wrc_dm_plugin.py` | wrc-demo | `DmBridgeArm` and `BridgeCamera` backends, registered at runtime |
| `claude_llm.py` | wrc-demo | Claude client for WRC's agent loop |
| `run_dm.py` | wrc-demo | Launcher: `--dashboard` or `--task` |
| `make_dm_urdf.py` | wrc-demo | Builds `assets/ReBot_Arm_DM_wrc.urdf` from Seeed's DM URDF |
| `install.sh` | shell | One-time setup checks and downloads |
| `calibrate_camera.py` | wrc-demo | Camera-to-base calibration from the printed sheet; writes a camera profile |
| `calib_sheet_A4.png`, `calibration_example.yaml` | – | The sheet to print and the input file to fill in |
| `calibration_selftest.py` | wrc-demo | `./START_WRC.sh dm-calib-test`: checks the calibration against the sim's true camera |
| `two_view.py` | wrc-demo | Object position, height and width from the outlines in one or two calibrated RGB cameras (no depth) |
| `measure_objects.py` | wrc-demo | `./START_WRC.sh dm-measure`: prints what the cameras measure; moves nothing |
| `known_objects_example.yaml` | – | Table of known object sizes for `--known-objects` |
| `two_view_selftest.py`, `camera_measure_wiring_test.py` | wrc-demo | `./START_WRC.sh dm-measure-test`: the maths on drawn scenes, and the wiring into the pipeline |
| `fake_real_arm.py` | teleoperation venv | Rehearses the bridge's real-arm path against a stand-in driver, no hardware |
| `../configs/arms/rebot_dm.yaml`, `../configs/cameras/dm_sim_overhead.yaml` | – | WRC profiles |

Why WRC's shipped `anthropic` profile is not used: it sends a `temperature`, which current Claude models reject.

## Limitations

- The sim camera is ideal: exact calibration, perfect depth, no noise. Real grasps will be less accurate than the 5–15 mm seen here.
- The detector is the weak point. On the sim image it finds the red block at 0.77 confidence, the yellow cylinder (as "yellow ball") at 0.67 and the blue block at only 0.25. Labels matter: "red cube" scored lower than "red block".
- WRC gives Claude only four useful tools. It cannot place at arbitrary coordinates, push, or rotate objects.
- WRC's instructions tell the LLM not to retry a failed pick, so one failure ends the task.
- After a failed place the object stays in the gripper. Send `open the gripper` or restart.

## License

© 2026 ferohatami, MIT License. The license text is in `wrc_dm_pipeline/LICENSE` in the team repository. It covers the files of this port (`local_dm/`, `local_mock/`, `START_WRC.sh`, `SETUP_REPORT.md`, the two `configs/` profiles), not WRC.
