import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = None


def build_project_paths(project_root=None):
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    else:
        project_root = Path(project_root).expanduser().resolve()

    return {
        "root": project_root,
        "notebooks": project_root / "notebooks" / "coursework",
        "results": project_root / "results" / "coursework",
        "plan": project_root / "results" / "coursework" / "robot_rr_plan.npz",
    }


PATHS = build_project_paths(project_root=PROJECT_ROOT)
PLAN_PATH = PATHS["plan"]

if not PLAN_PATH.exists():
    raise FileNotFoundError(
        f"No existe {PLAN_PATH}. "
        f"Run {PATHS['notebooks'] / 'robot_RR_spline.ipynb'} first to generate the plan."
    )

plan = np.load(PLAN_PATH)

TIME_GRID = plan["time"]

# El notebook usa la convencion del curso: angulo positivo eleva el eslabon.
# En este modelo de MuJoCo, el hinge alrededor de +y tiene el signo opuesto,
# asi que aqui se hace el cambio de signo al consumir el plan exportado.
Q_REF = -plan["q"]
QD_REF = -plan["qdot"]
QDD_REF = -plan["qddot"]
TAU_FF_REF = -plan["tau_ff"]

DESIRED_FINAL_TIP = plan["desired_final_tip"]
Q_START = -plan["q_start"]
Q_GOAL_PHYSICAL = -plan["q_goal_physical"]
Q_GOAL = -plan["q_goal"]
FINAL_ABSOLUTE_ORIENTATION = -float(plan["final_absolute_orientation"])

L1 = float(plan["L1"])
L2 = float(plan["L2"])
LC1 = float(plan["LC1"])
LC2 = float(plan["LC2"])
BASE_Z = float(plan["base_z"])
M1 = float(plan["M1"])
M2 = float(plan["M2"])
I1 = float(plan["I1"])
I2 = float(plan["I2"])
GRAVITY = float(plan["gravity"])
OBSTACLE_CENTER = plan["obstacle_center"]
OBSTACLE_HALF_SIZE = plan["obstacle_half_size"]
SIM_TIME = float(plan["sim_time"])
TIMESTEP = float(plan["dt"])


MUJOCO_XML = f"""
<mujoco model="robot_RR_con_obstaculo">
  <compiler angle="degree"/>
  <option gravity="0 0 -{GRAVITY}" timestep="{TIMESTEP}"/>

  <default>
    <joint damping="0.2" armature="0.0"/>
    <geom contype="1" conaffinity="1" friction="0.8 0.1 0.1" condim="3"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
  </asset>

  <worldbody>
    <geom name="floor" type="plane" material="grid" size="5 5 5" pos="0 0 -1.5"/>
    <light pos="0 0 5" directional="true" diffuse="0.9 0.9 0.9" specular="0.1 0.1 0.1" dir="0 0 -1" castshadow="true"/>

    <body name="base" pos="0 0 {BASE_Z}">
      <geom name="base_geom" type="box" size="0.04 0.04 0.04" rgba="0.3 0.3 0.3 1"/>

      <body name="link1" pos="0 0 0">
        <inertial pos="{LC1} 0 0" mass="{M1}" diaginertia="0.001 {I1} {I1}"/>
        <joint name="q1" type="hinge" axis="0 1 0" range="-360 360"/>
        <geom name="link1_geom"
              type="capsule"
              fromto="0 0 0 {L1} 0 0.0"
              size="0.025"
              rgba="0.8 0.4 0.4 1"/>

        <body name="link2" pos="{L1} 0 0">
          <inertial pos="{LC2} 0 0" mass="{M2}" diaginertia="0.001 {I2} {I2}"/>
          <joint name="q2" type="hinge" axis="0 1 0" range="-360 360"/>
          <geom name="link2_geom"
                type="capsule"
                fromto="0 0 0 {L2} 0 0"
                size="0.022"
                rgba="0.4 0.4 0.8 1"/>
        </body>
      </body>
    </body>

    <body name="obstacle" pos="{OBSTACLE_CENTER[0]} 0 {OBSTACLE_CENTER[1]}">
      <geom name="box_obstacle"
            type="box"
            size="{OBSTACLE_HALF_SIZE[0]} 0.08 {OBSTACLE_HALF_SIZE[1]}"
            rgba="0.2 0.8 0.2 1"/>
    </body>
  </worldbody>

  <actuator>
    <motor joint="q1" ctrlrange="-300 300" gear="1"/>
    <motor joint="q2" ctrlrange="-300 300" gear="1"/>
  </actuator>
</mujoco>
"""


def interp_vector(t, values):
    t_clamped = float(np.clip(t, TIME_GRID[0], TIME_GRID[-1]))
    return np.array(
        [np.interp(t_clamped, TIME_GRID, values[:, idx]) for idx in range(values.shape[1])],
        dtype=float,
    )


def forward_kinematics(q1, q2):
    elbow = np.array([
        L1 * np.cos(q1),
        BASE_Z - L1 * np.sin(q1),
    ])
    tip = elbow + np.array([
        L2 * np.cos(q1 + q2),
        -L2 * np.sin(q1 + q2),
    ])
    return elbow, tip


KP = np.array([220.0, 120.0])
KD = np.array([42.0, 24.0])
CTRL_LIMIT = 300.0


model = mujoco.MjModel.from_xml_string(MUJOCO_XML)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box_obstacle")


def geom_name(geom_id):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return name if name is not None else f"geom_{geom_id}"


def detect_box_contacts():
    contacts = []
    for idx in range(data.ncon):
        contact = data.contact[idx]
        if contact.geom1 == box_id or contact.geom2 == box_id:
            contacts.append(
                (
                    geom_name(contact.geom1),
                    geom_name(contact.geom2),
                    np.array(contact.pos),
                )
            )
    return contacts


collision_events = []
collision_active = False
max_tau = np.zeros(2)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        with viewer.lock():
            q_d = interp_vector(data.time, Q_REF)
            qd_d = interp_vector(data.time, QD_REF)
            tau_ff = interp_vector(data.time, TAU_FF_REF)

            q = data.qpos[:2].copy()
            qd = data.qvel[:2].copy()

            tau = tau_ff + KD * (qd_d - qd) + KP * (q_d - q)
            tau = np.clip(tau, -CTRL_LIMIT, CTRL_LIMIT)
            max_tau = np.maximum(max_tau, np.abs(tau))

            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

            contacts = detect_box_contacts()
            if contacts:
                model.geom_rgba[box_id] = np.array([1.0, 0.0, 0.0, 1.0])
            else:
                model.geom_rgba[box_id] = np.array([0.2, 0.8, 0.2, 1.0])

            if contacts and not collision_active:
                collision_active = True
                collision_events.append({
                    "time": float(data.time),
                    "contacts": contacts.copy(),
                })
            elif not contacts:
                collision_active = False

        viewer.sync()
        time.sleep(model.opt.timestep)


q_final = data.qpos[:2].copy()
elbow_2d, tip_2d = forward_kinematics(q_final[0], q_final[1])

elbow_pos = np.array([elbow_2d[0], 0.0, elbow_2d[1]])
tip_pos = np.array([tip_2d[0], 0.0, tip_2d[1]])

tip_behind_cube = tip_2d[0] > 0.60
elbow_above_cube = elbow_2d[1] > 0.30
if FINAL_ABSOLUTE_ORIENTATION >= 0.0:
    full_turn_completed = (q_final[0] + q_final[1]) > FINAL_ABSOLUTE_ORIENTATION - 0.05
else:
    full_turn_completed = (q_final[0] + q_final[1]) < FINAL_ABSOLUTE_ORIENTATION + 0.05
joint_error = Q_GOAL - q_final


print("\n===== COLLISION SUMMARY =====")
if not collision_events:
    print("No collision with the green box was detected.")
else:
    for idx, event in enumerate(collision_events, start=1):
        print(f"\nCollision #{idx} at t = {event['time']:.3f} s")
        for geom1, geom2, pos in event["contacts"]:
            print(f"  {geom1} <-> {geom2} at {pos}")


print("\n===== TRAJECTORY DESIGN =====")
print(f"Desired final tip position = {DESIRED_FINAL_TIP}")
print(f"IK physical goal          = {Q_GOAL_PHYSICAL}")
print(f"Wrapped goal for 360 deg  = {Q_GOAL}")
print(f"Initial joint state       = {Q_START}")


print("\n===== FINAL CONFIGURATION =====")
print(f"q1 = {q_final[0]:.3f} rad")
print(f"q2 = {q_final[1]:.3f} rad")
print(f"q1 + q2 = {q_final.sum():.3f} rad")
print(f"Joint tracking error = {joint_error}")
print(f"Elbow position = {elbow_pos}")
print(f"Tip position   = {tip_pos}")
print(f"Elbow above cube? {elbow_above_cube}")
print(f"Tip behind cube? {tip_behind_cube}")
print(f"Clockwise full turn completed? {full_turn_completed}")
print(f"Max |tau| = {max_tau}")
