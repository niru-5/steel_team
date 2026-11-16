#!/usr/bin/env python
"""Generate WRC's kinematic URDF for the reBot B601-DM (added by the hackathon port; NOT upstream).

Source: Seeed's official DM URDF (steel_team/teleoperation/assets/reBot-DevArm/
Rebot_Arm_description/DM/urdf/ReBot_Arm_DM.urdf, fetched by that folder's
scripts/setup_assets.sh). Two changes:

1. A `gripper_end` tool frame. WRC's planner and safety checks need one; the DM URDF ends at
   `end_link` (at the fingertips, tool along +x), so `gripper_end` sits 1.5 cm behind the
   fingertips -- the same "tip offset below TCP" convention WRC uses for the RS arm.
2. Joint ranges clipped to what the REAL arm accepts (REAL_SOFT_LIMITS_DEG in
   teleoperation/teleop/rebot_model.py, same as the team's follower.json). Seeed's URDF allows
   more (wrist roll +-180 deg, the arm +-90 deg). If WRC plans outside the real range, the
   command is clipped and the gripper arrives in the right place but turned wrong.

Only kinematics are used (pinocchio.buildModelFromUrdf), so mesh paths are irrelevant.
"""
import ast
import math
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TELEOP = Path(os.environ.get("STEEL_TEAM_TELEOP_DIR", HERE.parents[1] / "steel_team" / "teleoperation"))
SRC = TELEOP / "assets" / "reBot-DevArm" / "Rebot_Arm_description" / "DM" / "urdf" / "ReBot_Arm_DM.urdf"
OUT = HERE / "assets" / "ReBot_Arm_DM_wrc.urdf"
TIP_OFFSET_M = 0.015
# Fallback copy; the team's rebot_model.py is read first so the two cannot drift apart.
REAL_SOFT_LIMITS_DEG = {"joint1": (-150.0, 150.0), "joint2": (-200.0, 1.0), "joint3": (-200.0, 1.0),
                        "joint4": (-80.0, 90.0), "joint5": (-90.0, 90.0), "joint6": (-90.0, 90.0)}

EXTRA = f"""
  <!-- added by wrc/local_dm/make_dm_urdf.py: WRC tool frame -->
  <link name="gripper_end" />
  <joint name="gripper_end_joint" type="fixed">
    <origin xyz="{-TIP_OFFSET_M} 0 0" rpy="0 0 0" />
    <parent link="end_link" />
    <child link="gripper_end" />
  </joint>
</robot>"""


def real_limits() -> dict:
    model_py = TELEOP / "teleop" / "rebot_model.py"
    if model_py.exists():
        m = re.search(r"REAL_SOFT_LIMITS_DEG\s*=\s*(\{.*?\})", model_py.read_text(), re.S)
        if m:
            return ast.literal_eval(m.group(1))
    return REAL_SOFT_LIMITS_DEG


def clip_joint_limits(text: str, limits: dict) -> tuple[str, list[str]]:
    report = []
    for joint, (lo_deg, hi_deg) in limits.items():
        pat = re.compile(r'(<joint name="%s"[^>]*>.*?<limit\s+lower=")([-\d.eE+]+)("\s+upper=")([-\d.eE+]+)(")'
                         % re.escape(joint), re.S)
        m = pat.search(text)
        if m is None:
            raise SystemExit(f"could not find the limits of {joint} in {SRC}")
        lo0, hi0 = float(m.group(2)), float(m.group(4))
        lo, hi = max(lo0, math.radians(lo_deg)), min(hi0, math.radians(hi_deg))
        text = text[:m.start()] + f"{m.group(1)}{lo:.6f}{m.group(3)}{hi:.6f}{m.group(5)}" + text[m.end():]
        if (lo, hi) != (lo0, hi0):
            report.append(f"{joint}: {math.degrees(lo0):.0f}..{math.degrees(hi0):.0f} -> "
                          f"{math.degrees(lo):.0f}..{math.degrees(hi):.0f} deg")
    return text, report


if __name__ == "__main__":
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found: run scripts/setup_assets.sh in steel_team/teleoperation "
                         "(or set STEEL_TEAM_TELEOP_DIR)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = SRC.read_text()
    assert text.rstrip().endswith("</robot>"), "unexpected URDF layout"
    text, report = clip_joint_limits(text, real_limits())
    OUT.write_text(text.rstrip()[: -len("</robot>")] + EXTRA + "\n")
    print(f"wrote {OUT}")
    for line in report:
        print(f"  joint range clipped to the real arm: {line}")
