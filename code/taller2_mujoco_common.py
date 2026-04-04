from __future__ import annotations

import argparse
import os
import shutil
import site
import sys
import sysconfig
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import sympy as sp
from scipy.interpolate import BPoly
from scipy.optimize import fsolve, least_squares
from sympy.physics.mechanics import ReferenceFrame, dynamicsymbols


SCRIPT_PATH = Path(__file__).resolve()


def build_project_paths(project_root=None):
    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
    else:
        candidates = [
            Path.cwd().resolve(),
            SCRIPT_PATH.parent,
            SCRIPT_PATH.parent.parent,
            SCRIPT_PATH.parent.parent.parent,
        ]
        root = None
        for candidate in candidates:
            if (candidate / 'code').exists() and (candidate / 'datasets').exists():
                root = candidate
                break
        if root is None:
            root = SCRIPT_PATH.parent.parent

    return {
        'root': root,
        'code': root / 'code',
        'datasets': root / 'datasets',
    }


PATHS = build_project_paths()

MJPYTHON_BINARY_PATH = (
    Path(mujoco.__file__).resolve().parent
    / 'MuJoCo_(mjpython).app'
    / 'Contents'
    / 'MacOS'
    / 'mjpython'
)


def find_mjpython_launcher():
    candidates = []
    path_hit = shutil.which('mjpython')
    if path_hit is not None:
        candidates.append(Path(path_hit))

    candidates.append(Path(sys.executable).expanduser().resolve().with_name('mjpython'))

    scripts_dir = sysconfig.get_path('scripts')
    if scripts_dir:
        candidates.append(Path(scripts_dir).expanduser().resolve() / 'mjpython')

    user_base = site.getuserbase()
    if user_base:
        candidates.append(Path(user_base).expanduser().resolve() / 'bin' / 'mjpython')

    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate

    return None


BALL_DIAMETER = 27e-3
BALL_RADIUS = BALL_DIAMETER / 2.0
DEFAULT_DEFLECTOR_DRAW_OFFSET = np.array([np.hypot(17e-3, 38e-3), 0.0], dtype=float)
PICKUP_POINT_DRAW = np.array([120.0, 27.0]) * 1e-3
RELEASE_POINT_DRAW = np.array([165.0, 67.0]) * 1e-3
ACTUATED_BOUNDS_DEG = np.array([
    [90.0, -5.0],
    [180.0, 90.0],
])

SELECTED_SPRING = {
    'attach_name': 'C',
    'anchor_draw': np.array([0.0, -30.0]) * 1e-3,
}


LA, LB, LC, LD, LG, LH, LE, LE1, LF, LF1x, LF1y, L001x, L001y, LPx, LPy = sp.symbols(
    'LA, LB, LC, LD, LG, LH, LE, LE1, LF, LF1x, LF1y, L001x, L001y, LPx, LPy'
)

PARAMS = {
    LA: 80e-3,
    LB: 35e-3,
    LC: 80e-3,
    LD: 80e-3,
    LG: 80e-3,
    LH: 20e-3,
    LE: 35e-3,
    LE1: 80e-3,
    LF: 46e-3,
    LF1x: 36e-3,
    LF1y: 17e-3,
    L001x: 36e-3,
    L001y: 17e-3,
    LPx: DEFAULT_DEFLECTOR_DRAW_OFFSET[0],
    LPy: DEFAULT_DEFLECTOR_DRAW_OFFSET[1],
}

t_mech = dynamicsymbols._t
N = ReferenceFrame('N')
BODY_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
theta = dynamicsymbols('theta0:' + str(len(BODY_NAMES)))
FRAMES = {body: N.orientnew(body, 'Axis', [theta[i], N.z]) for i, body in enumerate(BODY_NAMES)}


def to_draw(point_model):
    return np.array([-point_model[0], point_model[1]], dtype=float)


POINTS = {}
POINTS['O'] = 0 * N.x
POINTS['O1'] = POINTS['O'] + L001x * N.x + L001y * N.y
POINTS['A'] = LA * FRAMES['A'].x
POINTS['B'] = LB * FRAMES['B'].x
POINTS['C'] = POINTS['B'] + LC * FRAMES['C'].x
POINTS['D'] = POINTS['B'] - LD * FRAMES['D'].x
POINTS['E'] = POINTS['C'] - LE * FRAMES['E'].x
POINTS['E1'] = POINTS['C'] - (LE + LE1) * FRAMES['E'].x
POINTS['H'] = POINTS['E1'] + LH * FRAMES['H'].x
POINTS['G'] = POINTS['H'] + LG * FRAMES['G'].x
POINTS['F'] = POINTS['G'] + LF * FRAMES['F'].x
# P is parameterized directly in the drawing frame so the deflector can stay parallel to the floor.
POINTS['P'] = POINTS['E1'] - LPx * N.x + LPy * N.y


def project_components(vector, frame):
    return [vector.dot(frame.x), vector.dot(frame.y), vector.dot(frame.z)]


eq1 = POINTS['E'] + LF1x * FRAMES['F'].x + LF1y * FRAMES['F'].y + LD * FRAMES['D'].x - POINTS['O1']
eq2 = POINTS['E'] - POINTS['A']
eq3 = LH * FRAMES['H'].x + LG * FRAMES['G'].x + (LF - LF1x) * FRAMES['F'].x - LF1y * FRAMES['F'].y - LE1 * FRAMES['E'].x

eq = project_components(eq1, N)[0:2] + project_components(eq2, N)[0:2] + project_components(eq3, N)[0:2]
eq = [expr.subs(PARAMS) for expr in eq]
eq_fun = sp.lambdify(theta, eq, 'numpy')

points_fun_model = {
    name: sp.lambdify(theta, [vector.dot(N.x).subs(PARAMS), vector.dot(N.y).subs(PARAMS)], 'numpy')
    for name, vector in POINTS.items()
}

z_seed = np.deg2rad([90.0, -90.0, 0.0, 0.0, 0.0, 100.0])

SEGMENT_PAIRS = [
    ('O', 'A'),
    ('O', 'B'),
    ('B', 'C'),
    ('B', 'D'),
    ('C', 'E1'),
    ('E1', 'H'),
    ('H', 'G'),
    ('G', 'F'),
    ('A', 'F'),
    ('O1', 'F'),
    ('E1', 'P'),
]

TREE_LINK_SPECS = [
    {'joint': 'qA', 'name': 'OA', 'start': 'O', 'end': 'A', 'radius': 0.0032, 'rgba': (0.55, 0.20, 0.60, 1.0)},
    {'joint': 'qB', 'name': 'OB', 'start': 'O', 'end': 'B', 'radius': 0.0028, 'rgba': (0.15, 0.15, 0.15, 1.0)},
    {'joint': 'qC', 'name': 'BC', 'start': 'B', 'end': 'C', 'radius': 0.0026, 'rgba': (0.90, 0.25, 0.25, 1.0)},
    {'joint': 'qD', 'name': 'BD', 'start': 'B', 'end': 'D', 'radius': 0.0026, 'rgba': (0.95, 0.55, 0.70, 1.0)},
    {'joint': 'qE', 'name': 'CE1', 'start': 'C', 'end': 'E1', 'radius': 0.0030, 'rgba': (0.15, 0.35, 0.80, 1.0)},
    {'joint': 'qH', 'name': 'E1H', 'start': 'E1', 'end': 'H', 'radius': 0.0024, 'rgba': (0.20, 0.70, 0.30, 1.0)},
    {'joint': 'qG', 'name': 'HG', 'start': 'H', 'end': 'G', 'radius': 0.0024, 'rgba': (0.85, 0.20, 0.20, 1.0)},
    {'joint': 'qF', 'name': 'GF', 'start': 'G', 'end': 'F', 'radius': 0.0024, 'rgba': (0.95, 0.55, 0.10, 1.0)},
]

AUX_VISUAL_SEGMENTS = [
    {'name': 'AF', 'start': 'A', 'end': 'F', 'radius': 0.0022, 'rgba': (0.95, 0.55, 0.10, 1.0)},
    {'name': 'O1F', 'start': 'O1', 'end': 'F', 'radius': 0.0022, 'rgba': (0.10, 0.35, 0.90, 1.0)},
    {'name': 'E1P', 'start': 'E1', 'end': 'P', 'radius': 0.0019, 'rgba': (0.45, 0.45, 0.45, 1.0)},
]

VISUAL_JOINTS = ['O', 'O1', 'A', 'B', 'C', 'D', 'E1', 'H', 'G', 'F', 'P']
JOINT_ORDER = ['qA', 'qB', 'qC', 'qD', 'qE', 'qH', 'qG', 'qF']


def FK(theta0_val, theta1_val, guess=None):
    if guess is None:
        guess = z_seed
    guess = np.asarray(guess, dtype=float)
    return fsolve(lambda dependent: eq_fun(theta0_val, theta1_val, *dependent), guess)


def mechanism_state(q_act, guess=None):
    q_act = np.asarray(q_act, dtype=float)
    dependent = FK(q_act[0], q_act[1], guess=guess)
    full_theta = np.array([q_act[0], q_act[1], *dependent], dtype=float)
    points_model = {
        name: np.array(points_fun_model[name](*full_theta), dtype=float)
        for name in points_fun_model
    }
    points_draw = {name: to_draw(points_model[name]) for name in points_model}
    return full_theta, points_model, points_draw, dependent


def build_waypoints():
    home_q_act = np.deg2rad([110.0, 30.0])
    home_full, home_points_model, home_points_draw, home_dependent = mechanism_state(home_q_act, guess=z_seed)

    def ik_actuated(target_draw, q_guess_deg):
        target_draw = np.asarray(target_draw, dtype=float)
        lower_bounds = np.deg2rad(ACTUATED_BOUNDS_DEG[0])
        upper_bounds = np.deg2rad(ACTUATED_BOUNDS_DEG[1])
        seed_guess = z_seed.copy()

        def residual(q_act_now):
            _, _, points_draw_now, _ = mechanism_state(q_act_now, guess=seed_guess)
            return points_draw_now['P'] - target_draw

        result = least_squares(
            residual,
            np.deg2rad(q_guess_deg),
            bounds=(lower_bounds, upper_bounds),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=200,
        )

        full_theta_now, points_model_now, points_draw_now, dependent_now = mechanism_state(result.x, guess=seed_guess)
        return {
            'q_act': result.x,
            'full_theta': full_theta_now,
            'points_model': points_model_now,
            'points_draw': points_draw_now,
            'dependent': dependent_now,
        }

    return {
        'home': {
            'q_act': home_q_act,
            'full_theta': home_full,
            'points_model': home_points_model,
            'points_draw': home_points_draw,
            'dependent': home_dependent,
        },
        'pre_pick': ik_actuated(np.array([120.0, 60.0]) * 1e-3, q_guess_deg=(100.0, 20.0)),
        'pickup': ik_actuated(PICKUP_POINT_DRAW, q_guess_deg=(100.0, 35.0)),
        'pre_release': ik_actuated(np.array([145.0, 72.0]) * 1e-3, q_guess_deg=(108.0, 2.0)),
        'release': ik_actuated(RELEASE_POINT_DRAW, q_guess_deg=(123.0, -2.0)),
    }


def build_piecewise_spline_trajectory():
    waypoints = build_waypoints()

    trajectory_segments = [
        {'name': 'home_to_pre_pick', 'start': 'home', 'end': 'pre_pick', 'duration': 1.4, 'payload_on': False},
        {'name': 'pre_pick_to_pickup', 'start': 'pre_pick', 'end': 'pickup', 'duration': 0.8, 'payload_on': False},
        {'name': 'pickup_dwell', 'start': 'pickup', 'end': 'pickup', 'duration': 0.4, 'payload_on': True},
        {'name': 'pickup_to_lift', 'start': 'pickup', 'end': 'pre_pick', 'duration': 0.9, 'payload_on': True},
        {'name': 'lift_to_pre_release', 'start': 'pre_pick', 'end': 'pre_release', 'duration': 2.0, 'payload_on': True},
        {'name': 'pre_release_to_release', 'start': 'pre_release', 'end': 'release', 'duration': 1.0, 'payload_on': True},
        {'name': 'release_dwell', 'start': 'release', 'end': 'release', 'duration': 0.4, 'payload_on': False},
    ]

    dt = 0.02
    segment_boundaries = [0.0]
    release_end_time = None
    for segment in trajectory_segments:
        segment_boundaries.append(segment_boundaries[-1] + float(segment['duration']))
        if segment['name'] == 'pre_release_to_release':
            release_end_time = segment_boundaries[-1]

    segment_boundaries = np.array(segment_boundaries, dtype=float)
    knot_names = [trajectory_segments[0]['start']] + [segment['end'] for segment in trajectory_segments]
    knot_q = np.array([waypoints[name]['q_act'] for name in knot_names], dtype=float)

    knot_data_theta0 = [[float(knot_q[i, 0]), 0.0, 0.0] for i in range(len(segment_boundaries))]
    knot_data_theta1 = [[float(knot_q[i, 1]), 0.0, 0.0] for i in range(len(segment_boundaries))]

    spline_theta0 = BPoly.from_derivatives(segment_boundaries.tolist(), knot_data_theta0)
    spline_theta1 = BPoly.from_derivatives(segment_boundaries.tolist(), knot_data_theta1)

    n_time = int(np.round(segment_boundaries[-1] / dt)) + 1
    time_grid = np.linspace(0.0, segment_boundaries[-1], n_time)
    q_act = np.column_stack([
        spline_theta0(time_grid),
        spline_theta1(time_grid),
    ])

    payload_mask = np.zeros(len(time_grid), dtype=float)
    phase_name = np.empty(len(time_grid), dtype=object)
    for i, segment in enumerate(trajectory_segments):
        start_time = segment_boundaries[i]
        end_time = segment_boundaries[i + 1]
        if i < len(trajectory_segments) - 1:
            mask = (time_grid >= start_time - 1e-12) & (time_grid < end_time - 1e-12)
        else:
            mask = (time_grid >= start_time - 1e-12) & (time_grid <= end_time + 1e-12)
        payload_mask[mask] = float(segment['payload_on'])
        phase_name[mask] = segment['name']

    full_theta = np.zeros((len(time_grid), len(theta)), dtype=float)
    points_history_draw = {name: np.zeros((len(time_grid), 2), dtype=float) for name in POINTS}

    z_guess = waypoints['home']['dependent'].copy()
    for i, q_now in enumerate(q_act):
        full_theta_now, _, points_draw_now, z_guess = mechanism_state(q_now, guess=z_guess)
        full_theta[i] = full_theta_now
        for name in POINTS:
            points_history_draw[name][i] = points_draw_now[name]

    ball_center_draw = np.zeros((len(time_grid), 2), dtype=float)
    pickup_contact_time = segment_boundaries[2]
    release_hold_start = release_end_time if release_end_time is not None else time_grid[-1]
    for i, time_now in enumerate(time_grid):
        if time_now < pickup_contact_time:
            ball_center_draw[i] = PICKUP_POINT_DRAW
        elif time_now <= release_hold_start:
            ball_center_draw[i] = points_history_draw['P'][i]
        else:
            ball_center_draw[i] = RELEASE_POINT_DRAW

    def segment_abs_angle(start_name, end_name):
        delta = points_history_draw[end_name] - points_history_draw[start_name]
        phi_draw = np.arctan2(delta[:, 1], delta[:, 0])  # angle in the x-z drawing plane
        return -np.unwrap(phi_draw)  # MuJoCo hinge around +y uses the opposite sign

    qA_abs = segment_abs_angle('O', 'A')
    qB_abs = segment_abs_angle('O', 'B')
    qC_abs = segment_abs_angle('B', 'C')
    qD_abs = segment_abs_angle('B', 'D')
    qE_abs = segment_abs_angle('C', 'E1')
    qH_abs = segment_abs_angle('E1', 'H')
    qG_abs = segment_abs_angle('H', 'G')
    qF_abs = segment_abs_angle('G', 'F')

    q_tree_history = np.column_stack([
        qA_abs,
        qB_abs,
        qC_abs - qB_abs,
        qD_abs - qB_abs,
        qE_abs - qC_abs,
        qH_abs - qE_abs,
        qG_abs - qH_abs,
        qF_abs - qG_abs,
    ])

    segment_lengths = {
        spec['name']: float(np.linalg.norm(points_history_draw[spec['end']][0] - points_history_draw[spec['start']][0]))
        for spec in TREE_LINK_SPECS + AUX_VISUAL_SEGMENTS
    }

    return {
        'time': time_grid,
        'dt': dt,
        'q_act': q_act,
        'waypoints': waypoints,
        'phase_name': phase_name,
        'payload_mask': payload_mask,
        'points_history_draw': points_history_draw,
        'ball_center_draw': ball_center_draw,
        'release_end_time': release_end_time,
        'segment_boundaries': segment_boundaries,
        'q_tree_history': q_tree_history,
        'segment_lengths': segment_lengths,
    }


def world_from_draw(point_draw):
    return np.array([point_draw[0], 0.0, point_draw[1]], dtype=float)


def capsule_pose_from_points(point_start_draw, point_end_draw):
    p0 = world_from_draw(point_start_draw)
    p1 = world_from_draw(point_end_draw)
    center = 0.5 * (p0 + p1)
    direction = p1 - p0
    length = np.linalg.norm(direction)
    angle_y = np.arctan2(direction[0], direction[2])
    quat = np.array([
        np.cos(0.5 * angle_y),
        0.0,
        np.sin(0.5 * angle_y),
        0.0,
    ], dtype=float)
    return center, quat, length


def build_mujoco_xml(trajectory):
    segment_lengths = trajectory['segment_lengths']
    o1_draw = trajectory['points_history_draw']['O1'][0]

    def rgba_text(spec):
        return ' '.join(f'{value:.6f}' for value in spec['rgba'])

    return f"""
<mujoco model='taller2_articulated_viewer'>
  <compiler angle='degree'/>
  <option gravity='0 0 0' timestep='0.01'/>
  <default>
    <joint damping='0.02' armature='0.0'/>
    <geom contype='0' conaffinity='0'/>
  </default>
  <visual>
    <global offwidth='800' offheight='600'/>
    <headlight diffuse='0.8 0.8 0.8' ambient='0.25 0.25 0.25' specular='0.1 0.1 0.1'/>
    <rgba haze='0.98 0.98 0.98 1'/>
  </visual>
  <worldbody>
    <light pos='0 0.2 0.35' dir='0 -0.3 -1' directional='true'/>
    <geom name='floor' type='plane' pos='0 0 0' size='1 1 0.05' rgba='0.95 0.95 0.95 1'/>
    <camera name='cam_main' pos='0.095 -0.39 0.100' xyaxes='1 0 0 0 0 1'/>

    <geom name='pickup_pedestal' type='box' pos='0.120 0 0.00575' size='0.015 0.015 0.00575' rgba='0.62 0.62 0.62 1'/>
    <body name='release_pedestal' pos='0.165 0 0.000'>
      <geom name='release_column' type='box' pos='0 0 0.028' size='0.0075 0.0075 0.028' rgba='0.62 0.62 0.62 1'/>
      <geom name='release_ring' type='cylinder' pos='0 0 0.067' size='0.018 0.0014' rgba='0.10 0.72 0.18 1'/>
    </body>

    <body name='base' pos='0 0 0'>
      <geom name='base_geom' type='box' pos='-0.006 0 0.0018' size='0.018 0.018 0.0018' rgba='0.22 0.22 0.22 1'/>
      <geom name='mark_O' type='sphere' pos='0 0 0' size='0.0032' rgba='0.05 0.05 0.05 1'/>
      <geom name='mark_O1' type='sphere' pos='{o1_draw[0]:.6f} 0 {o1_draw[1]:.6f}' size='0.0032' rgba='0.05 0.05 0.05 1'/>

      <body name='linkA' pos='0 0 0'>
        <joint name='qA' type='hinge' axis='0 1 0'/>
        <geom name='geom_OA' type='capsule' fromto='0 0 0 {segment_lengths["OA"]:.6f} 0 0' size='0.0032' rgba='{rgba_text(TREE_LINK_SPECS[0])}'/>
        <geom name='mark_A' type='sphere' pos='{segment_lengths["OA"]:.6f} 0 0' size='0.0028' rgba='0.10 0.10 0.10 1'/>
      </body>

      <body name='linkB' pos='0 0 0'>
        <joint name='qB' type='hinge' axis='0 1 0'/>
        <geom name='geom_OB' type='capsule' fromto='0 0 0 {segment_lengths["OB"]:.6f} 0 0' size='0.0028' rgba='{rgba_text(TREE_LINK_SPECS[1])}'/>
        <geom name='mark_B' type='sphere' pos='{segment_lengths["OB"]:.6f} 0 0' size='0.0028' rgba='0.10 0.10 0.10 1'/>

        <body name='linkC' pos='{segment_lengths["OB"]:.6f} 0 0'>
          <joint name='qC' type='hinge' axis='0 1 0'/>
          <geom name='geom_BC' type='capsule' fromto='0 0 0 {segment_lengths["BC"]:.6f} 0 0' size='0.0026' rgba='{rgba_text(TREE_LINK_SPECS[2])}'/>
          <geom name='mark_C' type='sphere' pos='{segment_lengths["BC"]:.6f} 0 0' size='0.0026' rgba='0.10 0.10 0.10 1'/>

          <body name='linkE' pos='{segment_lengths["BC"]:.6f} 0 0'>
            <joint name='qE' type='hinge' axis='0 1 0'/>
            <geom name='geom_CE1' type='capsule' fromto='0 0 0 {segment_lengths["CE1"]:.6f} 0 0' size='0.0030' rgba='{rgba_text(TREE_LINK_SPECS[4])}'/>
            <geom name='mark_E1' type='sphere' pos='{segment_lengths["CE1"]:.6f} 0 0' size='0.0026' rgba='0.10 0.10 0.10 1'/>

            <body name='linkH' pos='{segment_lengths["CE1"]:.6f} 0 0'>
              <joint name='qH' type='hinge' axis='0 1 0'/>
              <geom name='geom_E1H' type='capsule' fromto='0 0 0 {segment_lengths["E1H"]:.6f} 0 0' size='0.0024' rgba='{rgba_text(TREE_LINK_SPECS[5])}'/>
              <geom name='mark_H' type='sphere' pos='{segment_lengths["E1H"]:.6f} 0 0' size='0.0026' rgba='0.10 0.10 0.10 1'/>

              <body name='linkG' pos='{segment_lengths["E1H"]:.6f} 0 0'>
                <joint name='qG' type='hinge' axis='0 1 0'/>
                <geom name='geom_HG' type='capsule' fromto='0 0 0 {segment_lengths["HG"]:.6f} 0 0' size='0.0024' rgba='{rgba_text(TREE_LINK_SPECS[6])}'/>
                <geom name='mark_G' type='sphere' pos='{segment_lengths["HG"]:.6f} 0 0' size='0.0026' rgba='0.10 0.10 0.10 1'/>

                <body name='linkF' pos='{segment_lengths["HG"]:.6f} 0 0'>
                  <joint name='qF' type='hinge' axis='0 1 0'/>
                  <geom name='geom_GF' type='capsule' fromto='0 0 0 {segment_lengths["GF"]:.6f} 0 0' size='0.0024' rgba='{rgba_text(TREE_LINK_SPECS[7])}'/>
                  <geom name='mark_F' type='sphere' pos='{segment_lengths["GF"]:.6f} 0 0' size='0.0028' rgba='0.10 0.10 0.10 1'/>
                </body>
              </body>
            </body>
          </body>
        </body>

        <body name='linkD' pos='{segment_lengths["OB"]:.6f} 0 0'>
          <joint name='qD' type='hinge' axis='0 1 0'/>
          <geom name='geom_BD' type='capsule' fromto='0 0 0 {segment_lengths["BD"]:.6f} 0 0' size='0.0026' rgba='{rgba_text(TREE_LINK_SPECS[3])}'/>
          <geom name='mark_D' type='sphere' pos='{segment_lengths["BD"]:.6f} 0 0' size='0.0026' rgba='0.10 0.10 0.10 1'/>
        </body>
      </body>
    </body>

    <body name='aux_AF' pos='0 0 0'>
      <freejoint name='joint_aux_AF'/>
      <geom name='geom_aux_AF' type='capsule' size='0.0022 {0.5 * segment_lengths["AF"]:.6f}' rgba='{rgba_text(AUX_VISUAL_SEGMENTS[0])}'/>
    </body>

    <body name='aux_O1F' pos='0 0 0'>
      <freejoint name='joint_aux_O1F'/>
      <geom name='geom_aux_O1F' type='capsule' size='0.0022 {0.5 * segment_lengths["O1F"]:.6f}' rgba='{rgba_text(AUX_VISUAL_SEGMENTS[1])}'/>
    </body>

    <body name='aux_E1P' pos='0 0 0'>
      <freejoint name='joint_aux_E1P'/>
      <geom name='geom_aux_E1P' type='capsule' size='0.0019 {0.5 * segment_lengths["E1P"]:.6f}' rgba='{rgba_text(AUX_VISUAL_SEGMENTS[2])}'/>
    </body>

    <body name='mark_P_body' pos='0 0 0'>
      <freejoint name='joint_mark_P'/>
      <geom name='geom_mark_P' type='sphere' size='0.0032' rgba='0.85 0.15 0.15 1'/>
    </body>

    <body name='ball_visual' pos='0 0 0'>
      <freejoint name='joint_ball_visual'/>
      <geom name='geom_ball_visual' type='sphere' size='{BALL_RADIUS:.6f}' rgba='0.95 0.55 0.15 1'/>
    </body>

    <body name='spring_visual' pos='0 0 0'>
      <freejoint name='joint_spring_visual'/>
      <geom name='geom_spring_visual' type='capsule' size='0.001500 0.020000' rgba='0.85 0.85 0.15 1'/>
    </body>
  </worldbody>
</mujoco>
"""


def qpos_address(model, joint_name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return model.jnt_qposadr[joint_id]


def geom_id(model, geom_name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)


def set_free_body_pose(data, qpos_addr, center_world, quat):
    data.qpos[qpos_addr:qpos_addr + 3] = center_world
    data.qpos[qpos_addr + 3:qpos_addr + 7] = quat


def build_scene(with_spring=False):
    trajectory = build_piecewise_spline_trajectory()
    model = mujoco.MjModel.from_xml_string(build_mujoco_xml(trajectory))
    data = mujoco.MjData(model)

    scene = {
        'trajectory': trajectory,
        'model': model,
        'data': data,
        'with_spring': with_spring,
        'joint_qpos': {joint_name: qpos_address(model, joint_name) for joint_name in JOINT_ORDER},
        'aux_qpos': {spec['name']: qpos_address(model, f'joint_aux_{spec["name"]}') for spec in AUX_VISUAL_SEGMENTS},
        'aux_geom': {spec['name']: geom_id(model, f'geom_aux_{spec["name"]}') for spec in AUX_VISUAL_SEGMENTS},
        'mark_P_qpos': qpos_address(model, 'joint_mark_P'),
        'ball_qpos': qpos_address(model, 'joint_ball_visual'),
        'spring_qpos': qpos_address(model, 'joint_spring_visual'),
        'spring_geom': geom_id(model, 'geom_spring_visual'),
    }

    update_scene_pose(scene, 0)
    return scene


def should_relaunch_with_mjpython():
    if sys.platform != 'darwin':
        return False
    return getattr(mujoco.viewer, '_MJPYTHON', None) is None


def relaunch_with_mjpython_if_needed(args):
    if args.gif is not None:
        return
    if not should_relaunch_with_mjpython():
        return
    mjpython_launcher = find_mjpython_launcher()
    if mjpython_launcher is None:
        raise RuntimeError(
            'MuJoCo on macOS needs the mjpython launcher for the interactive viewer, '
            'but no executable mjpython wrapper was found. '
            f'Checked PATH, sys.executable sibling, sysconfig scripts, and user base. '
            f'Bundled binary path: {MJPYTHON_BINARY_PATH}'
        )
    os.execv(str(mjpython_launcher), [str(mjpython_launcher), *sys.argv])


def update_scene_pose(scene, sample_id):
    model = scene['model']
    data = scene['data']
    trajectory = scene['trajectory']
    points_history_draw = trajectory['points_history_draw']
    ball_center_draw = trajectory['ball_center_draw']
    q_tree = trajectory['q_tree_history'][sample_id]
    for joint_index, joint_name in enumerate(JOINT_ORDER):
        data.qpos[scene['joint_qpos'][joint_name]] = float(q_tree[joint_index])

    for spec in AUX_VISUAL_SEGMENTS:
        p0 = points_history_draw[spec['start']][sample_id]
        p1 = points_history_draw[spec['end']][sample_id]
        center_world, quat, length = capsule_pose_from_points(p0, p1)
        set_free_body_pose(data, scene['aux_qpos'][spec['name']], center_world, quat)
        model.geom_size[scene['aux_geom'][spec['name']], 1] = max(0.5 * length, 1e-4)

    p_center_world = world_from_draw(points_history_draw['P'][sample_id])
    set_free_body_pose(data, scene['mark_P_qpos'], p_center_world, np.array([1.0, 0.0, 0.0, 0.0]))

    ball_center_world = world_from_draw(ball_center_draw[sample_id])
    set_free_body_pose(data, scene['ball_qpos'], ball_center_world, np.array([1.0, 0.0, 0.0, 0.0]))

    if scene['with_spring']:
        spring_start_draw = SELECTED_SPRING['anchor_draw']
        spring_end_draw = points_history_draw[SELECTED_SPRING['attach_name']][sample_id]
        spring_center_world, spring_quat, spring_length = capsule_pose_from_points(spring_start_draw, spring_end_draw)
        set_free_body_pose(data, scene['spring_qpos'], spring_center_world, spring_quat)
        model.geom_rgba[scene['spring_geom']] = np.array([0.85, 0.85, 0.15, 1.0])
        model.geom_size[scene['spring_geom'], 1] = max(0.5 * spring_length, 1e-4)
    else:
        model.geom_rgba[scene['spring_geom']] = np.array([0.85, 0.85, 0.15, 0.0])

    data.time = float(trajectory['time'][sample_id])
    mujoco.mj_forward(model, data)


def print_summary(scene):
    trajectory = scene['trajectory']
    print(f'Project root: {PATHS["root"]}')
    print(f'Total motion time: {trajectory["time"][-1]:.2f} s')
    print(f'Release reached at: {trajectory["release_end_time"]:.2f} s')
    print(f'Spring visible: {scene["with_spring"]}')
    if scene['with_spring']:
        print(f'Spring attachment: {SELECTED_SPRING["attach_name"]}')
        print(f'Spring anchor [mm]: {1e3 * SELECTED_SPRING["anchor_draw"]}')


def export_gif(scene, output_path=None, n_frames=120):
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Pillow is required only for GIF export. Install it with 'pip install pillow' "
            "or run the script without '--gif' to open the MuJoCo viewer directly."
        ) from exc

    model = scene['model']
    data = scene['data']
    renderer = None

    try:
        renderer = mujoco.Renderer(model, height=600, width=800)
        frame_ids = np.linspace(0, len(scene['trajectory']['time']) - 1, n_frames, dtype=int)
        frames = []

        for sample_id in frame_ids:
            update_scene_pose(scene, int(sample_id))
            renderer.update_scene(data, camera='cam_main')
            frames.append(Image.fromarray(renderer.render()))

        if output_path is None:
            suffix = 'with_spring' if scene['with_spring'] else 'without_spring'
            output_path = PATHS['datasets'] / f'taller2_viewer_{suffix}.gif'
        else:
            output_path = Path(output_path).expanduser().resolve()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=50,
            loop=0,
        )
        print(f'GIF exported to: {output_path}')
        return output_path
    finally:
        if renderer is not None:
            renderer.close()


def run_viewer(scene, playback_rate=1.0, loop=True):
    dt = float(scene['trajectory']['dt']) / max(playback_rate, 1e-9)
    fixed_cam_id = mujoco.mj_name2id(scene['model'], mujoco.mjtObj.mjOBJ_CAMERA, 'cam_main')

    with mujoco.viewer.launch_passive(scene['model'], scene['data']) as viewer:
        if fixed_cam_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = fixed_cam_id

        sample_id = 0
        next_wall_time = time.time()

        while viewer.is_running():
            now = time.time()
            if now < next_wall_time:
                time.sleep(min(0.002, next_wall_time - now))
                continue

            with viewer.lock():
                update_scene_pose(scene, sample_id)
            viewer.sync()

            sample_id += 1
            if sample_id >= len(scene['trajectory']['time']):
                if loop:
                    sample_id = 0
                else:
                    break

            next_wall_time += dt


def parse_args(default_with_spring=False, argv=None):
    parser = argparse.ArgumentParser(
        description='Interactive MuJoCo viewer for the Taller 2 trajectory.',
    )
    parser.add_argument('--with-spring', action='store_true', default=default_with_spring, help='Show the spring in the MuJoCo scene.')
    parser.add_argument('--without-spring', action='store_true', help='Force the scene to hide the spring.')
    parser.add_argument('--playback-rate', type=float, default=1.0, help='Playback speed multiplier for the viewer.')
    parser.add_argument('--once', action='store_true', help='Play the motion once and exit.')
    parser.add_argument('--summary', action='store_true', help='Print the computed scene summary before launching or exporting.')
    parser.add_argument('--gif', type=str, default=None, help='Export the same scene as a GIF instead of opening the interactive viewer.')
    parser.add_argument('--frames', type=int, default=120, help='Number of frames when exporting a GIF.')
    return parser.parse_args(argv)


def main(default_with_spring=False, argv=None):
    args = parse_args(default_with_spring=default_with_spring, argv=argv)
    relaunch_with_mjpython_if_needed(args)
    with_spring = args.with_spring and not args.without_spring
    scene = build_scene(with_spring=with_spring)

    if args.summary:
        print_summary(scene)

    if args.gif is not None:
        export_gif(scene, output_path=args.gif, n_frames=args.frames)
        return 0

    run_viewer(scene, playback_rate=args.playback_rate, loop=not args.once)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
