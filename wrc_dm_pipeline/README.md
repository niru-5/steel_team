# Prompt-to-action pipeline for our reBot B601-DM

You type what you want, for example *"put the red block on the blue block"*. An AI (Claude) looks at the camera picture, decides what to do, and the robot arm does it.

It is built on [WRC](https://github.com/TheMoonAstronaut/wrc), an open-source robot pipeline made for the RS version of our arm. This folder adapts it to our **DM** arm.

> **Latest version (fourth push, 2026-11-15): no depth camera needed.** Our webcams give no distance, so on the real robot the can and the cup were placed and sized wrongly and the grasp missed. The pipeline now measures each object's position, height and width from its outline in our two fixed scene cameras. In the simulator with depth switched off: positions within 3 mm, 8 of 8 random pick-and-place trials into the bin; the old method got 0 of 4. **Not tested on the real cameras or the real arm.** Both scene cameras must be calibrated first. Details and how to use it: [`overlay/local_dm/README.md`](overlay/local_dm/README.md), first section. The third push (same day) switched the simulator to the team's real-workspace scene.

## How it works

1. **You type a task** in a web page.
2. **The camera** takes a picture of the table.
3. **An object detector** finds the things on the table and where they are.
4. **Claude** gets the picture and the list of objects, and picks an action, such as "pick up the red block and place it on the blue block".
5. **WRC plans the motion**: where to grab the object and how each joint has to move.
6. **A safety check** looks at every small step of the motion. The arm must stay inside its allowed area, above the table, and under a speed limit. If a step fails the check, the arm stops.
7. **The arm moves**: it reaches down, closes the gripper, carries the object and lets go.
8. **Claude reports** what happened, including when something failed.

```
you type a task → Claude (sees the camera) → WRC (plans + safety check) → arm
                                                                          ├─ simulator of our DM arm   ← works today
                                                                          └─ the real DM arm           ← not tested yet
```

## What works today

- **In the simulation of our real workspace: yes, for the cases tested.** Claude put the can into the near bin compartment and the cup into the far one (checked against the simulator's own object positions). Of 8 random start positions, 5 worked at first; the 3 failures were fixed and re-run one by one. The full random set was not repeated after the fix, so there is no success rate yet.
- **On the old demo table (three blocks): yes.** Claude stacked blocks, moved objects and described the scene, 5 to 15 mm from the goal. `WRC_DM_SCENE=demo ./START_WRC.sh dm` starts it.
- **No depth camera: handled with two scene cameras (new, simulator only).** With depth switched off in the simulator, objects are measured from their outlines in the corner and far cameras: 8 of 8 random trials worked (keyword commands). Check the real cameras first with `./START_WRC.sh dm-measure --camera A --camera2 B --table-z 0.0`, which prints what they measure and moves nothing. For the real arm the bin and wall positions still have to be configured by hand.
- **On the real arm: not tested.** The connection uses the same driver calls as our `teleoperation/go_ready.py`, but nobody has run this pipeline on the hardware. Start slowly and follow the checklist (link below).
- **Camera calibration: tool included, tested in the simulator only.** You print one sheet, measure two points with a ruler, and run one command.

## Try it in the simulator

You need conda, `uv`, and this repo's `teleoperation` folder set up (`./scripts/setup_assets.sh && uv sync`). For the real-workspace scene the repo must contain `simulation/` (it is on `main`); `uv sync` installs it.

```bash
bash wrc_dm_pipeline/setup_wrc.sh          # downloads WRC next to this repo and adds our files
cd ../wrc
PYTHONNOUSERSITE=1 conda env create -f environment.yml
bash local_dm/install.sh                   # checks everything and says what is missing
echo 'ANTHROPIC_API_KEY=your-key' > .env   # your own Claude API key (never commit it)
./START_WRC.sh dm                          # opens the 3D simulator and the web page
```

Then open http://127.0.0.1:8090/ and type a task **in that web page** (not in the 3D window, which is display only). Things to try:

- `Put the red can in the bin`
- `Now the paper cup goes in the other compartment`
- `What objects can you see on the table right now?`

`./START_WRC.sh dm-shuffle` moves the objects to new random places, so you can see that the robot finds them with its camera.

Without an API key it still runs, but only fixed commands work, like `pick and place the red block` or `go home`.

## Good to know

- The arm cannot reach everywhere on the table. Far objects are grabbed with the gripper leaning forward; the furthest grasp tested was 46 cm from the base. Objects out of reach are reported, not attempted blindly.
- The detector only finds objects whose **names it was given** (`--classes "cup,red cup,white box"`). It calls our plain red can "red cup" and the bin "white box". Change the names for other objects.
- Each task costs about 4 US cents in Claude API calls and takes 3 Claude steps of about 2 seconds each.

## What is in this folder

| Path | What it is |
|---|---|
| `setup_wrc.sh` | Downloads WRC and copies our files into it |
| `LICENSE` | License for this folder (MIT, © 2026 ferohatami) |
| `overlay/local_dm/` | Our code: the link between WRC and the DM arm (simulated or real), the Claude connection, camera calibration |
| `overlay/START_WRC.sh` | One script to start everything |
| `overlay/configs/` | Settings for the DM arm and the simulator camera |
| `overlay/local_mock/` | Earlier experiments without physics (not needed for the DM arm) |

## More detail

- **Full guide, test results, real-arm checklist, camera calibration:** [`overlay/local_dm/README.md`](overlay/local_dm/README.md)
- **How the WRC environment was set up (including the macOS fix):** [`overlay/SETUP_REPORT.md`](overlay/SETUP_REPORT.md)

## License

Everything in this folder (`wrc_dm_pipeline/`) is © 2026 ferohatami, released under the MIT License: see [`LICENSE`](LICENSE). The license covers this folder only. WRC itself is downloaded separately and keeps its own MIT license; the rest of this repository belongs to its own authors.
