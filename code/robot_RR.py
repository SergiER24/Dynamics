import mujoco
import mujoco.viewer
import numpy as np
import time
 
XML = r"""
<mujoco model="robot_RR_con_obstaculo">
 
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81" timestep="0.002"/>
 
  <default>
    <joint damping="1" armature="0.01"/>
    <geom contype="1" conaffinity="1" friction="0.8 0.1 0.1" condim="3"/>
  </default>
 
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
  </asset>
 
  <worldbody>
    <!-- Piso invisible -->
    <geom name="floor" type="plane" material="grid" size="5 5 5" pos="0 0 -1.5"/>
    <light pos="0 0 5" directional="true" diffuse="0.9 0.9 0.9" specular="0.1 0.1 0.1" dir="0 0 -1" castshadow="true"/>
 
    <!-- Base -->
    <body name="base" pos="0 0 0.05">
      <geom name="base_geom" type="box" size="0.04 0.04 0.04" rgba="0.3 0.3 0.3 1"/>
 
      <!-- Link 1 -->
      <body name="link1" pos="0 0 0">
        <joint name="q1" type="hinge" axis="0 1 0" range="-360 360"/>
        <geom name="link1_geom"
              type="capsule"
              fromto="0 0 0 0.7 0 0.0"
              size="0.025"
              rgba="0.8 0.4 0.4 1"/>
 
        <!-- Link 2 -->
        <body name="link2" pos="0.7 0 0">
          <joint name="q2" type="hinge" axis="0 1 0" range="-360 360"/>
          <geom name="link2_geom"
                type="capsule"
                fromto="0 0 0 0.7 0 0"
                size="0.022"
                rgba="0.4 0.4 0.8 1"/>
        </body>
      </body>
    </body>
 
    <!-- Caja -->
    <body name="obstacle" pos="0.5 0 0.2">
      <geom name="box_obstacle"
            type="box"
            size="0.10 0.08 0.10"
            rgba="0.2 0.8 0.2 1"/>
    </body>
  </worldbody>
 
  <actuator>
    <!-- Motor actuators: accept raw torque commands -->
    <motor joint="q1" ctrlrange="-200 200" gear="1"/>
    <motor joint="q2" ctrlrange="-200 200" gear="1"/>
  </actuator>
 
</mujoco>
"""
 
# ─────────────────────────────────────────────
#  PID Controller
# ─────────────────────────────────────────────
class PIDController:
    """
    Discrete PID with:
      - Anti-windup clamp on the integral term
      - Derivative on measurement (not on error) to avoid derivative kick
        when the setpoint changes abruptly
    """
    def __init__(self, kp, ki, kd, dt,
                 u_min=-200.0, u_max=200.0,
                 i_min=-50.0,  i_max=50.0):
        self.kp    = kp
        self.ki    = ki
        self.kd    = kd
        self.dt    = dt
        self.u_min = u_min
        self.u_max = u_max
        self.i_min = i_min   # anti-windup limits on integral accumulator
        self.i_max = i_max
 
        self._integral    = 0.0
        self._prev_meas   = None   # for derivative-on-measurement
 
    def reset(self):
        self._integral  = 0.0
        self._prev_meas = None
 
    def compute(self, setpoint, measurement):
        error = setpoint - measurement
 
        # --- Integral with anti-windup clamp ---
        self._integral += error * self.dt
        self._integral  = np.clip(self._integral, self.i_min, self.i_max)
 
        # --- Derivative on measurement (avoids kick on setpoint step) ---
        if self._prev_meas is None:
            self._prev_meas = measurement
        d_term = -(measurement - self._prev_meas) / self.dt
        self._prev_meas = measurement
 
        u = self.kp * error + self.ki * self._integral + self.kd * d_term
        return float(np.clip(u, self.u_min, self.u_max))
 
 
# ─────────────────────────────────────────────
#  Gravity compensation (feedforward)
#  For a 2-DOF planar arm in the XZ plane:
#    τ1 = (m1*lc1 + m2*l1)*g*cos(q1) + m2*lc2*g*cos(q1+q2)
#    τ2 = m2*lc2*g*cos(q1+q2)
#  We approximate with MuJoCo's own bias forces (data.qfrc_bias).
# ─────────────────────────────────────────────
def gravity_compensation(data):
    """
    Returns the torques needed to hold the arm against gravity,
    read directly from MuJoCo's computed bias force vector.
    """
    return data.qfrc_bias[:2].copy()   # [τ_q1, τ_q2]
 
 
# ─────────────────────────────────────────────
#  Trajectory  
# ─────────────────────────────────────────────
def trayectoria(t):
    if t < 2.0:
        q1_d = 0.0
        q2_d = 0.0
    elif t < 6.0:
        s = (t - 2.0) / 4.0
        s = 3*s**2 - 2*s**3          # smooth cubic profile
        q1_d = -0.8 * s
        q2_d =  1.1 * s
    else:
        q1_d = -0.8 + 0.20*np.sin(1.2*(t - 6.0))
        q2_d =  1.1 + 0.15*np.sin(1.8*(t - 6.0))
    return q1_d, q2_d
 
 
# ─────────────────────────────────────────────
#  Model & initial state
# ─────────────────────────────────────────────
model = mujoco.MjModel.from_xml_string(XML)
data  = mujoco.MjData(model)
 
dt = model.opt.timestep   # 0.002 s
 
data.qpos[0] = 0.0
data.qpos[1] = 0.0
data.ctrl[0] = 0.0
data.ctrl[1] = 0.0
mujoco.mj_forward(model, data)
 
# ─────────────────────────────────────────────
#  PID gains  — tune these to taste
#
#  Start conservatively:
#    kp  : proportional – main tracking gain
#    ki  : integral     – eliminates steady-state error (gravity drift)
#    kd  : derivative   – damps oscillation
# ─────────────────────────────────────────────
pid1 = PIDController(kp=120.0, ki=25.0, kd=8.0,  dt=dt)
pid2 = PIDController(kp=80.0,  ki=20.0, kd=5.0,  dt=dt)
 
 
# ─────────────────────────────────────────────
#  Collision helpers (unchanged)
# ─────────────────────────────────────────────
box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box_obstacle")
 
def geom_name(geom_id):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return name if name is not None else f"geom_{geom_id}"
 
def detect_box_contacts(model, data, box_id):
    contacts = []
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 == box_id or c.geom2 == box_id:
            contacts.append(
                (geom_name(c.geom1), geom_name(c.geom2), np.array(c.pos))
            )
    return contacts
 
 
# ─────────────────────────────────────────────
#  Simulation loop
# ─────────────────────────────────────────────
collision_events  = []
collision_active  = False
 
with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
 
    while viewer.is_running() and time.time() - start < 200:
        with viewer.lock():
            t = data.time
 
            # 1. Desired joint angles from trajectory
            q1_d, q2_d = trayectoria(t)
 
            # 2. Current measured joint angles
            q1 = data.qpos[0]
            q2 = data.qpos[1]
 
            # 3. PID output (feedback torque)
            tau1_fb = pid1.compute(q1_d, q1)
            tau2_fb = pid2.compute(q2_d, q2)
 
            # 4. Gravity-compensation feedforward torque
            #    (this is what cancels the "falling" effect)
            tau_grav = gravity_compensation(data)
 
            # 5. Total torque command = PID feedback + gravity feedforward
            data.ctrl[0] = tau1_fb + tau_grav[0]
            data.ctrl[1] = tau2_fb + tau_grav[1]
 
            mujoco.mj_step(model, data)
 
            # 6. Collision detection & visualisation
            contacts = detect_box_contacts(model, data, box_id)
 
            if len(contacts) > 0:
                model.geom_rgba[box_id] = np.array([1.0, 0.0, 0.0, 1.0])
            else:
                model.geom_rgba[box_id] = np.array([0.2, 0.8, 0.2, 1.0])
 
            if len(contacts) > 0 and not collision_active:
                collision_active = True
                collision_events.append({
                    "time": float(t),
                    "contacts": contacts.copy()
                })
            elif len(contacts) == 0:
                collision_active = False
 
        viewer.sync()
        time.sleep(dt)
 
# ─────────────────────────────────────────────
#  Report
# ─────────────────────────────────────────────
print("\n===== COLLISION SUMMARY =====")
if not collision_events:
    print("No collision with the green box was detected.")
else:
    for k, event in enumerate(collision_events, start=1):
        print(f"\nCollision #{k} at t = {event['time']:.3f} s")
        for g1, g2, pos in event["contacts"]:
            print(f"  {g1} <-> {g2} at {pos}")