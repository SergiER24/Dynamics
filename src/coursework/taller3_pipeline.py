from __future__ import annotations

import csv
import json
import math
import os
import shutil
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import differential_evolution, least_squares


def resolve_project_root() -> Path:
    file_var = globals().get("__file__")
    candidates = []
    if file_var:
        candidates.append(Path(file_var).resolve().parents[2])
    candidates.append(Path.cwd().resolve())
    candidates.append(Path.cwd().resolve().parent)

    for candidate in candidates:
        if (candidate / "src" / "coursework").exists() and (candidate / "data" / "raw").exists():
            return candidate
    return candidates[0]


PROJECT_ROOT = resolve_project_root()
CODE_DIR = PROJECT_ROOT / "src" / "coursework"
DATASETS_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "results" / "coursework" / "workshop_3"

def external_input(env_var: str, filename: str) -> Path:
    configured = os.environ.get(env_var)
    if configured:
        return Path(configured).expanduser()
    return PROJECT_ROOT / "external_inputs" / filename


SOURCE_VIDEO = external_input("DINAMICA_TALLER3_SOURCE_VIDEO", "source_video.mp4")
SOURCE_SYNTHESIS = external_input("DINAMICA_TALLER3_SOURCE_SYNTHESIS", "Sintesis.ppsx")
SOURCE_PATH_PLANNING = external_input("DINAMICA_TALLER3_SOURCE_PATH_PLANNING", "PathPlanning.pdf")
SOURCE_OPT = external_input("DINAMICA_TALLER3_SOURCE_OPT", "Optimizacion.pdf")
SOURCE_HUFFARD = external_input("DINAMICA_TALLER3_SOURCE_HUFFARD", "3697.pdf")
SOURCE_DOCX = external_input("DINAMICA_TALLER3_SOURCE_DOCX", "Taller 3 Dinamica.docx")

LOCAL_VIDEO = OUTPUT_DIR / "octopus_bipedal_reference.mp4"

BODY_TRACK_CSV = OUTPUT_DIR / "body_tracking_all_frames.csv"
DISTAL_TRACK_CSV = OUTPUT_DIR / "distal_tip_tracked_points.csv"
DISTAL_CYCLE_CSV = OUTPUT_DIR / "reference_trajectory_bl.csv"
PHASE_CSV = OUTPUT_DIR / "phase_sequence.csv"
ANGLES_FULL_CSV = OUTPUT_DIR / "equivalent_joint_angles_full_cycle_deg.csv"
ANGLES_KEY_CSV = OUTPUT_DIR / "equivalent_joint_angles_key_events_deg.csv"
FOURBAR_CSV = OUTPUT_DIR / "fourbar_dimensions.csv"
MECHANISM_TRACE_CSV = OUTPUT_DIR / "trajectory_overlay_trace.csv"
FOURBAR_STATES_CSV = OUTPUT_DIR / "fourbar_full_cycle_states.csv"
SIXBAR_CSV = OUTPUT_DIR / "sixbar_dimensions.csv"
SIXBAR_TRACE_CSV = OUTPUT_DIR / "sixbar_trajectory_trace.csv"
MUJOCO_TRACE_CSV = OUTPUT_DIR / "mujoco_tip_cycle.csv"
SUMMARY_JSON = OUTPUT_DIR / "taller3_summary.json"
MUJOCO_GIF = OUTPUT_DIR / "taller3_mujoco_static.gif"
MOTIONGEN_EIGHTBAR_CSV = OUTPUT_DIR / "motiongen_8bar_parameters.csv"
MOTIONGEN_EIGHTBAR_STATES_CSV = OUTPUT_DIR / "motiongen_8bar_states.csv"
SIXBAR_MUJOCO_GIF = OUTPUT_DIR / "sixbar_mechanism_mujoco.gif"
DESIGN_CYCLE_CSV = OUTPUT_DIR / "design_trajectory_bl.csv"
FOURBAR_COLLISION_CSV = OUTPUT_DIR / "fourbar_collision_report.csv"

FIGURE_SPECS = {
    "video_sheet": OUTPUT_DIR / "video_keyframes.png",
    "tracked_points_keyframes": OUTPUT_DIR / "tracked_points_keyframes.png",
    "body_path": OUTPUT_DIR / "body_path_over_video.png",
    "ciclogram": OUTPUT_DIR / "reference_ciclogram.png",
    "phase": OUTPUT_DIR / "phase_sequence.png",
    "angles": OUTPUT_DIR / "equivalent_joint_angles.png",
    "overlay": OUTPUT_DIR / "trajectory_overlay_vs_fourbar.png",
    "mechanism": OUTPUT_DIR / "mechanism_snapshots.png",
    "fourbar_full_cycle": OUTPUT_DIR / "fourbar_full_cycle.png",
    "fourbar_clearance": OUTPUT_DIR / "fourbar_clearance_vs_phase.png",
    "fourbar_cable_concept": OUTPUT_DIR / "fourbar_cable_tpu_concept.png",
    "sixbar_overlay": OUTPUT_DIR / "trajectory_overlay_vs_sixbar.png",
    "sixbar_snapshots": OUTPUT_DIR / "sixbar_snapshots.png",
    "sixbar_module": OUTPUT_DIR / "single_sixbar_module.png",
    "dual_architecture": OUTPUT_DIR / "dual_sixbar_architecture.png",
    "mujoco_match": OUTPUT_DIR / "mujoco_target_vs_simulation.png",
    "mujoco_frame": OUTPUT_DIR / "taller3_mujoco_static.png",
    "motiongen_mujoco_frame": OUTPUT_DIR / "motiongen_8bar_mujoco.png",
    "sixbar_mujoco_frame": OUTPUT_DIR / "sixbar_mechanism_mujoco_frame0.png",
}

COURSE_METADATA = {
    "university": "Universidad de los Andes",
    "department": "Departamento de Ingeniería Mecánica",
    "course": "Dinámica de Maquinaria",
    "title": "Taller 3",
    "animal": "Pulpo (Abdopus aculeatus)",
    "date": "20 de mayo de 2026",
    "professor": "Jonathan Camargo",
}

GROUP_MEMBERS = [
    ("Sergio Emanuel Ropero", "202120446"),
    ("David Alejandro Puentes Aldana", "202022517"),
    ("Alberto Luis Alario Caicedo", "201711829"),
    ("Sebastian Coy", "202220612"),
]

TRACKER_INIT_BBOX = (102, 68, 88, 96)
DISTAL_TRACK_FRAMES = np.arange(0, 101, 10, dtype=int)

# Manual digitization over the key frames.
# `PRIMARY_TIP_TRACK_PX` is the tip used to build the reference trajectory.
# `SECONDARY_TIP_TRACK_PX` marks the second support tentacle when visible.
PRIMARY_TIP_TRACK_PX = np.array(
    [
        [116.0, 161.0],
        [111.0, 160.0],
        [110.0, 151.0],
        [121.0, 144.0],
        [124.0, 164.0],
        [132.0, 177.0],
        [136.0, 186.0],
        [140.0, 176.0],
        [140.0, 186.0],
        [146.0, 171.0],
        [168.0, 179.0],
    ],
    dtype=float,
)

SECONDARY_TIP_TRACK_PX = np.array(
    [
        [153.0, 161.0],
        [157.0, 163.0],
        [161.0, 165.0],
        [159.0, 162.0],
        [158.0, 161.0],
        [154.0, 159.0],
        [151.0, 158.0],
        [149.0, 155.0],
        [155.0, 154.0],
        [179.0, 148.0],
        [201.0, 150.0],
    ],
    dtype=float,
)

DISTAL_TRACK_PHASE = np.linspace(0.0, 10.0 / 11.0, len(PRIMARY_TIP_TRACK_PX))

PHASE_MARKERS = {
    "contacto": 0.00,
    "apoyo": 0.32,
    "despegue": 0.62,
    "vuelo": 0.82,
}

PHASE_DESCRIPTION = {
    "contacto": "El extremo distal entra nuevamente en contacto con el sustrato.",
    "apoyo": "La extremidad permanece cerca del fondo y empuja hacia atrás.",
    "despegue": "La punta deja el sustrato e inicia el retorno.",
    "vuelo": "La extremidad avanza elevada antes del siguiente contacto.",
}

RIGHT_STANCE_START = 0.00
RIGHT_STANCE_END = 0.62
LEFT_STANCE_START = 0.50
LEFT_STANCE_END = 1.00
LEFT_STANCE_WRAP_END = 0.12

EQUIVALENT_BASE_BL = np.array([0.00, -0.08], dtype=float)
EQUIVALENT_LINKS_BL = np.array([0.17, 0.17, 0.18], dtype=float)
MUJOCO_ARM_LINKS_BL = np.array([0.19, 0.18, 0.18, 0.17, 0.17, 0.16, 0.15], dtype=float)
MUJOCO_TPU_LINKS_BL = np.array([0.090, 0.082, 0.074, 0.066, 0.058, 0.052], dtype=float)
MUJOCO_CABLE_MARKERS = 12
MUJOCO_BASE_A_BL = np.array([-0.22, -0.14], dtype=float)
MUJOCO_BASE_B_BL = np.array([0.08, -0.14], dtype=float)
MUJOCO_BODY_X_SCALE = 0.88
MUJOCO_BODY_Z_SCALE = 0.42
MUJOCO_BODY_Z_OFFSET = 0.16
FOURBAR_MARGIN_MIN_BL = 0.04
FOURBAR_PROJECTED_CLEARANCE_MIN_BL = 0.015
SIXBAR_MARGIN_MIN_BL = 0.03
DESIGN_FOURIER_ORDER = 3

# Approximate node coordinates digitized from the MotionGen screenshot and
# mapped to the BL frame used in the trajectory plots.
MOTIONGEN_8BAR_POINTS_BL = {
    "A": np.array([-0.8200, -0.3100], dtype=float),
    "B": np.array([-0.5600, -0.3288], dtype=float),
    "C": np.array([-0.2986, -0.3896], dtype=float),
    "D": np.array([-0.8021, -0.6801], dtype=float),
    "E": np.array([-0.6583, -0.6060], dtype=float),
    "F": np.array([-0.3731, -0.7355], dtype=float),
    "G": np.array([-0.2288, -0.8600], dtype=float),
    "H": np.array([-0.7463, -0.8000], dtype=float),
    "I": np.array([-0.4651, -0.8924], dtype=float),
    "J": np.array([-0.4189, -1.1095], dtype=float),
    "K": np.array([-0.3438, -1.1790], dtype=float),
}

# Bodies reconstructed from the screenshot. Ternary links are represented by
# the set of joints that must preserve pairwise distances.
MOTIONGEN_8BAR_BODIES = {
    "top_plate": ("A", "B", "C"),
    "left_input": ("D", "B"),
    "cross_link": ("A", "E", "I"),
    "center_plate": ("D", "E", "F"),
    "output_link": ("F", "G"),
    "lower_link": ("H", "K"),
    "lower_plate": ("I", "J", "K"),
}

# Visible edges for rendering in 2D and MuJoCo.
MOTIONGEN_8BAR_VIEW_EDGES = [
    ("A", "B"),
    ("B", "C"),
    ("A", "C"),
    ("D", "B"),
    ("A", "E"),
    ("E", "I"),
    ("A", "I"),
    ("D", "E"),
    ("E", "F"),
    ("D", "F"),
    ("F", "G"),
    ("H", "K"),
    ("I", "J"),
    ("J", "K"),
    ("I", "K"),
]

MOTIONGEN_FIXED_NODES = ("D", "H")
MOTIONGEN_OUTPUT_NODE = "G"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def copy_inputs() -> None:
    ensure_output_dir()
    if LOCAL_VIDEO.exists():
        return
    if SOURCE_VIDEO.exists():
        try:
            shutil.copy2(SOURCE_VIDEO, LOCAL_VIDEO)
        except PermissionError:
            if not LOCAL_VIDEO.exists():
                raise


def save_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_video_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No fue posible abrir el video {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def video_metadata(frames: list[np.ndarray], fps: float) -> dict[str, float]:
    if not frames:
        raise ValueError("El video no contiene fotogramas.")
    height, width = frames[0].shape[:2]
    return {
        "fps": fps,
        "n_frames": len(frames),
        "width_px": width,
        "height_px": height,
        "duration_s": len(frames) / fps,
    }


def track_body(frames: list[np.ndarray], fps: float) -> dict[str, np.ndarray]:
    tracker_factory = (
        cv2.TrackerCSRT_create
        if hasattr(cv2, "TrackerCSRT_create")
        else cv2.legacy.TrackerCSRT_create
    )
    tracker = tracker_factory()
    tracker.init(frames[0], TRACKER_INIT_BBOX)

    bbox_rows = []
    bbox = TRACKER_INIT_BBOX
    for frame_idx, frame in enumerate(frames):
        if frame_idx > 0:
            _, bbox = tracker.update(frame)
        x, y, w, h = map(float, bbox)
        bbox_rows.append([frame_idx, frame_idx / fps, x, y, w, h, x + 0.5 * w, y + 0.5 * h, h])

    track = np.asarray(bbox_rows, dtype=float)
    return {
        "rows": track,
        "frame": track[:, 0].astype(int),
        "time_s": track[:, 1],
        "bbox_x": track[:, 2],
        "bbox_y": track[:, 3],
        "bbox_w": track[:, 4],
        "bbox_h": track[:, 5],
        "center_x": track[:, 6],
        "center_y": track[:, 7],
        "body_length_px": track[:, 8],
    }


def build_distal_tracking_records(body_track: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rows = []
    for sample_idx, frame_idx in enumerate(DISTAL_TRACK_FRAMES):
        center_x = float(body_track["center_x"][frame_idx])
        center_y = float(body_track["center_y"][frame_idx])
        body_length_px = float(body_track["body_length_px"][frame_idx])
        phase_fraction = float(DISTAL_TRACK_PHASE[sample_idx])
        primary_tip_x_px, primary_tip_y_px = PRIMARY_TIP_TRACK_PX[sample_idx]
        secondary_tip_x_px, secondary_tip_y_px = SECONDARY_TIP_TRACK_PX[sample_idx]
        primary_tip_x_bl = (primary_tip_x_px - center_x) / body_length_px
        primary_tip_y_bl = -(primary_tip_y_px - center_y) / body_length_px
        secondary_tip_x_bl = (secondary_tip_x_px - center_x) / body_length_px
        secondary_tip_y_bl = -(secondary_tip_y_px - center_y) / body_length_px
        rows.append(
            [
                int(frame_idx),
                float(body_track["time_s"][frame_idx]),
                phase_fraction,
                center_x,
                center_y,
                body_length_px,
                float(primary_tip_x_px),
                float(primary_tip_y_px),
                float(primary_tip_x_bl),
                float(primary_tip_y_bl),
                float(secondary_tip_x_px),
                float(secondary_tip_y_px),
                float(secondary_tip_x_bl),
                float(secondary_tip_y_bl),
            ]
        )

    records = np.asarray(rows, dtype=float)
    return {
        "rows": records,
        "frame": records[:, 0].astype(int),
        "time_s": records[:, 1],
        "phase_fraction": records[:, 2],
        "body_center_x_px": records[:, 3],
        "body_center_y_px": records[:, 4],
        "body_length_px": records[:, 5],
        "primary_tip_x_px": records[:, 6],
        "primary_tip_y_px": records[:, 7],
        "primary_tip_x_bl": records[:, 8],
        "primary_tip_y_bl": records[:, 9],
        "secondary_tip_x_px": records[:, 10],
        "secondary_tip_y_px": records[:, 11],
        "secondary_tip_x_bl": records[:, 12],
        "secondary_tip_y_bl": records[:, 13],
    }


def build_reference_cycle(distal_records: dict[str, np.ndarray], n_samples: int = 121) -> dict[str, np.ndarray]:
    control_phase = distal_records["phase_fraction"].copy()
    control_xy = np.column_stack(
        [
            distal_records["primary_tip_x_bl"],
            distal_records["primary_tip_y_bl"],
        ]
    )
    closed_phase = np.append(control_phase, 1.0)
    closed_xy = np.vstack([control_xy, control_xy[0]])
    fine_phase = np.linspace(0.0, 1.0, n_samples)
    x_spline = CubicSpline(closed_phase, closed_xy[:, 0], bc_type="periodic")
    y_spline = CubicSpline(closed_phase, closed_xy[:, 1], bc_type="periodic")
    xy = np.column_stack([x_spline(fine_phase), y_spline(fine_phase)])
    dxy = np.column_stack([x_spline(fine_phase, 1), y_spline(fine_phase, 1)])
    ddxy = np.column_stack([x_spline(fine_phase, 2), y_spline(fine_phase, 2)])
    return {
        "phase": fine_phase,
        "xy": xy,
        "dxy": dxy,
        "ddxy": ddxy,
        "control_phase": control_phase,
        "control_xy": control_xy,
    }


def fourier_design_matrix(phase: np.ndarray, order: int) -> np.ndarray:
    phase = np.asarray(phase, dtype=float)
    columns = [np.ones_like(phase)]
    for harmonic in range(1, order + 1):
        angle = 2.0 * math.pi * harmonic * phase
        columns.append(np.cos(angle))
        columns.append(np.sin(angle))
    return np.column_stack(columns)


def evaluate_fourier_series(
    phase: np.ndarray,
    coeffs: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = np.asarray(phase, dtype=float)
    value = np.full_like(phase, float(coeffs[0]), dtype=float)
    first = np.zeros_like(phase, dtype=float)
    second = np.zeros_like(phase, dtype=float)
    idx = 1
    for harmonic in range(1, order + 1):
        a_n = float(coeffs[idx])
        b_n = float(coeffs[idx + 1])
        angle = 2.0 * math.pi * harmonic * phase
        omega = 2.0 * math.pi * harmonic
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        value += a_n * cos_angle + b_n * sin_angle
        first += (-omega * a_n) * sin_angle + (omega * b_n) * cos_angle
        second += (-(omega**2) * a_n) * cos_angle + (-(omega**2) * b_n) * sin_angle
        idx += 2
    return value, first, second


def build_design_cycle(
    reference_cycle: dict[str, np.ndarray],
    order: int = DESIGN_FOURIER_ORDER,
) -> dict[str, np.ndarray | float | int]:
    fit_phase = np.asarray(reference_cycle["phase"][:-1], dtype=float)
    fit_xy = np.asarray(reference_cycle["xy"][:-1], dtype=float)
    design_matrix = fourier_design_matrix(fit_phase, order)
    coeffs_x, *_ = np.linalg.lstsq(design_matrix, fit_xy[:, 0], rcond=None)
    coeffs_y, *_ = np.linalg.lstsq(design_matrix, fit_xy[:, 1], rcond=None)

    phase = np.asarray(reference_cycle["phase"], dtype=float)
    x, dx, ddx = evaluate_fourier_series(phase, coeffs_x, order)
    y, dy, ddy = evaluate_fourier_series(phase, coeffs_y, order)
    xy = np.column_stack([x, y])
    dxy = np.column_stack([dx, dy])
    ddxy = np.column_stack([ddx, ddy])
    rms_to_reference = float(
        np.sqrt(np.mean(np.sum((xy[:-1] - fit_xy) ** 2, axis=1)))
    )
    return {
        "phase": phase,
        "xy": xy,
        "dxy": dxy,
        "ddxy": ddxy,
        "order": int(order),
        "coeffs_x": np.asarray(coeffs_x, dtype=float),
        "coeffs_y": np.asarray(coeffs_y, dtype=float),
        "rms_to_reference_bl": rms_to_reference,
    }


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    return np.mod(phase, 1.0)


def is_stance_right(phase: np.ndarray) -> np.ndarray:
    return (phase >= RIGHT_STANCE_START) & (phase <= RIGHT_STANCE_END)


def is_stance_left(phase: np.ndarray) -> np.ndarray:
    wrapped = wrap_phase(phase)
    return ((wrapped >= LEFT_STANCE_START) & (wrapped <= LEFT_STANCE_END)) | (
        wrapped <= LEFT_STANCE_WRAP_END
    )


def equivalent_leg_fk(q: np.ndarray, base_xy: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    joints = [np.asarray(base_xy, dtype=float)]
    theta = 0.0
    point = np.asarray(base_xy, dtype=float)
    for angle, length in zip(q, lengths):
        theta += angle
        point = point + length * np.array([math.cos(theta), math.sin(theta)], dtype=float)
        joints.append(point.copy())
    return np.asarray(joints)


def solve_equivalent_angles(reference_cycle: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    xy = reference_cycle["xy"]
    dxy = reference_cycle["dxy"]
    phase = reference_cycle["phase"]
    q_seed = np.deg2rad(np.array([-95.0, 55.0, 25.0]))
    all_q = []
    all_joints = []

    for point, tangent in zip(xy, dxy):
        desired_orientation = math.atan2(tangent[1], tangent[0] + 1e-9)

        def residual(q: np.ndarray) -> np.ndarray:
            joints = equivalent_leg_fk(q, EQUIVALENT_BASE_BL, EQUIVALENT_LINKS_BL)
            tip = joints[-1]
            phi = float(np.sum(q))
            return np.array(
                [
                    tip[0] - point[0],
                    tip[1] - point[1],
                    0.12 * math.atan2(math.sin(phi - desired_orientation), math.cos(phi - desired_orientation)),
                ],
                dtype=float,
            )

        res = least_squares(
            residual,
            q_seed,
            bounds=(np.deg2rad([-170.0, -170.0, -170.0]), np.deg2rad([170.0, 170.0, 170.0])),
            max_nfev=4000,
        )
        q_seed = res.x
        joints = equivalent_leg_fk(res.x, EQUIVALENT_BASE_BL, EQUIVALENT_LINKS_BL)
        all_q.append(res.x.copy())
        all_joints.append(joints.copy())

    angles_rad = np.asarray(all_q)
    return {
        "phase": phase,
        "angles_rad": angles_rad,
        "angles_deg": np.rad2deg(angles_rad),
        "joints": np.asarray(all_joints),
    }


def key_phase_rows(angle_solution: dict[str, np.ndarray], reference_cycle: dict[str, np.ndarray]) -> list[dict[str, float | str]]:
    rows = []
    for phase_name, phase_value in PHASE_MARKERS.items():
        idx = int(np.argmin(np.abs(reference_cycle["phase"] - phase_value)))
        q_deg = angle_solution["angles_deg"][idx]
        xy = reference_cycle["xy"][idx]
        rows.append(
            {
                "fase": phase_name,
                "phase_fraction": float(reference_cycle["phase"][idx]),
                "x_bl": float(xy[0]),
                "y_bl": float(xy[1]),
                "theta1_deg": float(q_deg[0]),
                "theta2_deg": float(q_deg[1]),
                "theta3_deg": float(q_deg[2]),
            }
        )
    return rows


def fourbar_state(params: np.ndarray, phase_value: float, branch: int = 1) -> dict[str, np.ndarray | float] | None:
    x0, y0, phi, d, a, b, c, px, py, theta_offset = params
    o2 = np.array([x0, y0], dtype=float)
    o4 = o2 + d * np.array([math.cos(phi), math.sin(phi)], dtype=float)
    theta2 = theta_offset + (2.0 * math.pi * phase_value)
    a_point = o2 + a * np.array([math.cos(theta2), math.sin(theta2)], dtype=float)
    diff = o4 - a_point
    dist = float(np.linalg.norm(diff))
    if dist < 1e-9 or dist > b + c or dist < abs(b - c):
        return None

    ex = diff / dist
    ey = np.array([-ex[1], ex[0]], dtype=float)
    x = (b**2 - c**2 + dist**2) / (2.0 * dist)
    h_sq = b**2 - x**2
    if h_sq < 0.0:
        return None
    h = math.sqrt(max(h_sq, 0.0))
    b_point = a_point + x * ex + branch * h * ey
    theta3 = math.atan2(b_point[1] - a_point[1], b_point[0] - a_point[0])
    p_point = a_point + np.array(
        [
            px * math.cos(theta3) - py * math.sin(theta3),
            px * math.sin(theta3) + py * math.cos(theta3),
        ],
        dtype=float,
    )
    return {
        "o2": o2,
        "a": a_point,
        "b": b_point,
        "o4": o4,
        "p": p_point,
        "theta2": theta2,
        "theta3": theta3,
        "dist": dist,
        "upper_margin": (b + c) - dist,
        "lower_margin": dist - abs(b - c),
    }


def fourbar_trace(params: np.ndarray, phase: np.ndarray, branch: int = 1) -> tuple[np.ndarray, bool, dict[str, np.ndarray]]:
    trace = []
    valid = True
    upper_margins = []
    lower_margins = []
    theta3_values = []
    b_points = []
    min_clearances = []
    collision_flags = []

    for s in phase:
        state = fourbar_state(params, float(s), branch=branch)
        if state is None:
            valid = False
            trace.append([np.nan, np.nan])
            upper_margins.append(np.nan)
            lower_margins.append(np.nan)
            theta3_values.append(np.nan)
            b_points.append([np.nan, np.nan])
            min_clearances.append(np.nan)
            collision_flags.append(np.nan)
            continue
        trace.append(np.asarray(state["p"], dtype=float))
        upper_margins.append(float(state["upper_margin"]))
        lower_margins.append(float(state["lower_margin"]))
        theta3_values.append(float(state["theta3"]))
        b_points.append(np.asarray(state["b"], dtype=float))
        segments = fourbar_link_segments(state)
        clearance_values = []
        collision_here = False
        for idx in range(len(segments)):
            for jdx in range(idx + 1, len(segments)):
                first = segments[idx]
                second = segments[jdx]
                if first["joints"] & second["joints"]:
                    continue
                clearance = segment_distance(
                    np.asarray(first["p1"], dtype=float),
                    np.asarray(first["p2"], dtype=float),
                    np.asarray(second["p1"], dtype=float),
                    np.asarray(second["p2"], dtype=float),
                )
                clearance_values.append(float(clearance))
                collision_here = collision_here or (clearance < 1e-6)
        min_clearances.append(float(min(clearance_values)) if clearance_values else np.nan)
        collision_flags.append(float(collision_here))

    diagnostics = {
        "upper_margin": np.asarray(upper_margins, dtype=float),
        "lower_margin": np.asarray(lower_margins, dtype=float),
        "theta3": np.asarray(theta3_values, dtype=float),
        "b_points": np.asarray(b_points, dtype=float),
        "min_clearance": np.asarray(min_clearances, dtype=float),
        "collision_flag": np.asarray(collision_flags, dtype=float),
    }
    return np.asarray(trace, dtype=float), valid, diagnostics


def synthesize_fourbar(target_xy: np.ndarray, phase: np.ndarray) -> dict[str, np.ndarray | float | int]:
    bounds = [
        (-0.4, 0.4),    # x0
        (-0.6, 0.1),    # y0
        (-math.pi, math.pi),  # phi
        (0.10, 0.90),   # d
        (0.08, 0.70),   # a
        (0.08, 0.90),   # b
        (0.08, 0.90),   # c
        (-0.90, 0.90),  # px
        (-0.90, 0.90),  # py
        (-math.pi, math.pi),  # theta_offset
    ]

    best_solution = None
    best_error = np.inf

    for branch in (-1, 1):
        def objective(vector: np.ndarray) -> float:
            trace, valid, diagnostics = fourbar_trace(vector, phase, branch=branch)
            if (not valid) or np.isnan(trace).any():
                return 1e3
            rms = float(np.sqrt(np.mean(np.sum((trace - target_xy) ** 2, axis=1))))
            compactness = 0.02 * float(np.sum(np.abs(vector[3:7])))
            toggle_penalty = 40.0 * float(
                np.mean(np.maximum(0.0, FOURBAR_MARGIN_MIN_BL - diagnostics["upper_margin"]) ** 2)
                + np.mean(np.maximum(0.0, FOURBAR_MARGIN_MIN_BL - diagnostics["lower_margin"]) ** 2)
            )
            collision_penalty = 30.0 * float(
                np.mean(
                    np.maximum(
                        0.0,
                        FOURBAR_PROJECTED_CLEARANCE_MIN_BL - diagnostics["min_clearance"],
                    )
                    ** 2
                )
            )
            return rms + compactness + toggle_penalty + collision_penalty

        de_res = differential_evolution(
            objective,
            bounds=bounds,
            seed=7,
            popsize=12,
            maxiter=50,
            polish=False,
            tol=1e-4,
        )

        def residual(vector: np.ndarray) -> np.ndarray:
            trace, valid, diagnostics = fourbar_trace(vector, phase, branch=branch)
            if (not valid) or np.isnan(trace).any():
                return np.ones(target_xy.size + 2 * len(phase), dtype=float) * 10.0
            penalty_upper = 4.0 * np.maximum(
                0.0,
                FOURBAR_MARGIN_MIN_BL - diagnostics["upper_margin"],
            )
            penalty_lower = 4.0 * np.maximum(
                0.0,
                FOURBAR_MARGIN_MIN_BL - diagnostics["lower_margin"],
            )
            penalty_collision = 4.0 * np.maximum(
                0.0,
                FOURBAR_PROJECTED_CLEARANCE_MIN_BL - diagnostics["min_clearance"],
            )
            return np.concatenate(
                [
                    (trace - target_xy).ravel(),
                    penalty_upper,
                    penalty_lower,
                    penalty_collision,
                ]
            )

        lb = np.array([item[0] for item in bounds], dtype=float)
        ub = np.array([item[1] for item in bounds], dtype=float)
        lsq_res = least_squares(
            residual,
            de_res.x,
            bounds=(lb, ub),
            max_nfev=6000,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )

        trace, valid, diagnostics = fourbar_trace(lsq_res.x, phase, branch=branch)
        if not valid:
            continue
        rms = float(np.sqrt(np.mean(np.sum((trace - target_xy) ** 2, axis=1))))
        if rms < best_error:
            best_error = rms
            best_solution = {
                "params": lsq_res.x.copy(),
                "trace": trace.copy(),
                "branch": branch,
                "rms_error_bl": rms,
                "closure_error_bl": float(np.linalg.norm(trace[0] - trace[-1])),
                "min_upper_margin_bl": float(np.min(diagnostics["upper_margin"])),
                "min_lower_margin_bl": float(np.min(diagnostics["lower_margin"])),
            }

    if best_solution is None:
        raise RuntimeError("No se encontró una síntesis de cuatro barras válida.")
    return best_solution


def sixbar_state(
    params: np.ndarray,
    phase_value: float,
    fourbar_branch: int = 1,
    dyad_branch: int = 1,
) -> dict[str, np.ndarray | float] | None:
    fourbar_params = np.asarray(params[:10], dtype=float)
    o6x, o6y, pq_len, o6q_len = [float(value) for value in params[10:14]]
    base_state = fourbar_state(fourbar_params, phase_value, branch=fourbar_branch)
    if base_state is None:
        return None

    p_point = np.asarray(base_state["p"], dtype=float)
    o6 = np.array([o6x, o6y], dtype=float)
    diff = o6 - p_point
    dist = float(np.linalg.norm(diff))
    if dist < 1e-9 or dist > pq_len + o6q_len or dist < abs(pq_len - o6q_len):
        return None

    ex = diff / dist
    ey = np.array([-ex[1], ex[0]], dtype=float)
    x = (pq_len**2 - o6q_len**2 + dist**2) / (2.0 * dist)
    h_sq = pq_len**2 - x**2
    if h_sq < 0.0:
        return None
    h = math.sqrt(max(h_sq, 0.0))
    q_point = p_point + x * ex + dyad_branch * h * ey

    theta_pq = math.atan2(q_point[1] - p_point[1], q_point[0] - p_point[0])
    theta_o6q = math.atan2(q_point[1] - o6[1], q_point[0] - o6[0])
    state = dict(base_state)
    state.update(
        {
            "o6": o6,
            "q": q_point,
            "theta_pq": theta_pq,
            "theta_o6q": theta_o6q,
            "pq_upper_margin": (pq_len + o6q_len) - dist,
            "pq_lower_margin": dist - abs(pq_len - o6q_len),
            "pq_len": pq_len,
            "o6q_len": o6q_len,
            "p_o6_dist": dist,
        }
    )
    return state


def sixbar_trace(
    params: np.ndarray,
    phase: np.ndarray,
    fourbar_branch: int = 1,
    dyad_branch: int = 1,
) -> tuple[np.ndarray, np.ndarray, bool, dict[str, np.ndarray]]:
    p_trace = []
    q_trace = []
    valid = True
    fourbar_upper = []
    fourbar_lower = []
    dyad_upper = []
    dyad_lower = []
    q_points = []

    for s in phase:
        state = sixbar_state(params, float(s), fourbar_branch=fourbar_branch, dyad_branch=dyad_branch)
        if state is None:
            valid = False
            p_trace.append([np.nan, np.nan])
            q_trace.append([np.nan, np.nan])
            q_points.append([np.nan, np.nan])
            fourbar_upper.append(np.nan)
            fourbar_lower.append(np.nan)
            dyad_upper.append(np.nan)
            dyad_lower.append(np.nan)
            continue

        p_trace.append(np.asarray(state["p"], dtype=float))
        q_trace.append(np.asarray(state["q"], dtype=float))
        q_points.append(np.asarray(state["q"], dtype=float))
        fourbar_upper.append(float(state["upper_margin"]))
        fourbar_lower.append(float(state["lower_margin"]))
        dyad_upper.append(float(state["pq_upper_margin"]))
        dyad_lower.append(float(state["pq_lower_margin"]))

    diagnostics = {
        "fourbar_upper_margin": np.asarray(fourbar_upper, dtype=float),
        "fourbar_lower_margin": np.asarray(fourbar_lower, dtype=float),
        "dyad_upper_margin": np.asarray(dyad_upper, dtype=float),
        "dyad_lower_margin": np.asarray(dyad_lower, dtype=float),
        "q_points": np.asarray(q_points, dtype=float),
    }
    return np.asarray(p_trace, dtype=float), np.asarray(q_trace, dtype=float), valid, diagnostics


def synthesize_sixbar(
    target_xy: np.ndarray,
    phase: np.ndarray,
    fourbar_solution: dict[str, np.ndarray | float | int],
) -> dict[str, np.ndarray | float | int]:
    target_min = np.min(target_xy, axis=0)
    target_max = np.max(target_xy, axis=0)
    base_params = np.asarray(fourbar_solution["params"], dtype=float)
    base_branch = int(fourbar_solution["branch"])
    phase_global = phase[::3]
    target_global = target_xy[::3]

    base_bounds = [
        (-0.4, 0.4),
        (-0.6, 0.1),
        (-math.pi, math.pi),
        (0.10, 0.90),
        (0.08, 0.70),
        (0.08, 0.90),
        (0.08, 0.90),
        (-0.90, 0.90),
        (-0.90, 0.90),
        (-math.pi, math.pi),
    ]
    extra_bounds = [
        (float(target_min[0] - 0.55), float(target_max[0] + 0.55)),
        (float(target_min[1] - 0.60), float(target_max[1] + 0.35)),
        (0.05, 0.80),
        (0.05, 0.80),
    ]
    all_bounds = base_bounds + extra_bounds
    lb = np.array([item[0] for item in all_bounds], dtype=float)
    ub = np.array([item[1] for item in all_bounds], dtype=float)

    best_solution = None
    best_error = np.inf

    for fourbar_branch in (-1, 1):
        for dyad_branch in (-1, 1):
            def objective_global(vector: np.ndarray) -> float:
                _, q_trace, valid, diagnostics = sixbar_trace(
                    vector,
                    phase_global,
                    fourbar_branch=fourbar_branch,
                    dyad_branch=dyad_branch,
                )
                if (not valid) or np.isnan(q_trace).any():
                    return 1e3
                rms = float(np.sqrt(np.mean(np.sum((q_trace - target_global) ** 2, axis=1))))
                compactness = 0.01 * float(np.sum(np.abs(vector[3:7])) + vector[12] + vector[13])
                penalty = 28.0 * float(
                    np.mean(np.maximum(0.0, FOURBAR_MARGIN_MIN_BL - diagnostics["fourbar_upper_margin"]) ** 2)
                    + np.mean(np.maximum(0.0, FOURBAR_MARGIN_MIN_BL - diagnostics["fourbar_lower_margin"]) ** 2)
                    + np.mean(np.maximum(0.0, SIXBAR_MARGIN_MIN_BL - diagnostics["dyad_upper_margin"]) ** 2)
                    + np.mean(np.maximum(0.0, SIXBAR_MARGIN_MIN_BL - diagnostics["dyad_lower_margin"]) ** 2)
                )
                return rms + compactness + penalty

            de_res = differential_evolution(
                objective_global,
                bounds=all_bounds,
                seed=31 + 11 * (fourbar_branch + 1) + 7 * (dyad_branch + 1),
                popsize=8,
                maxiter=35,
                polish=False,
                tol=1e-4,
            )

            initial_candidates = [de_res.x]
            if fourbar_branch == base_branch:
                def objective_extra(extra_vector: np.ndarray) -> float:
                    params = np.concatenate([base_params, np.asarray(extra_vector, dtype=float)])
                    _, q_trace, valid, diagnostics = sixbar_trace(
                        params,
                        phase_global,
                        fourbar_branch=base_branch,
                        dyad_branch=dyad_branch,
                    )
                    if (not valid) or np.isnan(q_trace).any():
                        return 1e3
                    rms = float(np.sqrt(np.mean(np.sum((q_trace - target_global) ** 2, axis=1))))
                    compactness = 0.01 * float(np.sum(np.abs(extra_vector[2:])))
                    penalty = 30.0 * float(
                        np.mean(np.maximum(0.0, SIXBAR_MARGIN_MIN_BL - diagnostics["dyad_upper_margin"]) ** 2)
                        + np.mean(np.maximum(0.0, SIXBAR_MARGIN_MIN_BL - diagnostics["dyad_lower_margin"]) ** 2)
                    )
                    return rms + compactness + penalty

                de_extra = differential_evolution(
                    objective_extra,
                    bounds=extra_bounds,
                    seed=19 + 5 * (dyad_branch + 1),
                    popsize=10,
                    maxiter=35,
                    polish=False,
                    tol=1e-4,
                )
                initial_candidates.append(np.concatenate([base_params, de_extra.x]))

            def residual(vector: np.ndarray) -> np.ndarray:
                _, q_trace, valid, diagnostics = sixbar_trace(
                    vector,
                    phase,
                    fourbar_branch=fourbar_branch,
                    dyad_branch=dyad_branch,
                )
                if (not valid) or np.isnan(q_trace).any():
                    return np.ones(target_xy.size + 4 * len(phase), dtype=float) * 10.0
                penalty_fb_upper = 3.0 * np.maximum(
                    0.0,
                    FOURBAR_MARGIN_MIN_BL - diagnostics["fourbar_upper_margin"],
                )
                penalty_fb_lower = 3.0 * np.maximum(
                    0.0,
                    FOURBAR_MARGIN_MIN_BL - diagnostics["fourbar_lower_margin"],
                )
                penalty_dyad_upper = 3.0 * np.maximum(
                    0.0,
                    SIXBAR_MARGIN_MIN_BL - diagnostics["dyad_upper_margin"],
                )
                penalty_dyad_lower = 3.0 * np.maximum(
                    0.0,
                    SIXBAR_MARGIN_MIN_BL - diagnostics["dyad_lower_margin"],
                )
                return np.concatenate(
                    [
                        (q_trace - target_xy).ravel(),
                        penalty_fb_upper,
                        penalty_fb_lower,
                        penalty_dyad_upper,
                        penalty_dyad_lower,
                    ]
                )

            for x_init in initial_candidates:
                lsq_res = least_squares(
                    residual,
                    x_init,
                    bounds=(lb, ub),
                    max_nfev=9000,
                    xtol=1e-10,
                    ftol=1e-10,
                    gtol=1e-10,
                )

                p_trace, q_trace, valid, diagnostics = sixbar_trace(
                    lsq_res.x,
                    phase,
                    fourbar_branch=fourbar_branch,
                    dyad_branch=dyad_branch,
                )
                if not valid:
                    continue
                rms = float(np.sqrt(np.mean(np.sum((q_trace - target_xy) ** 2, axis=1))))
                if rms < best_error:
                    best_error = rms
                    best_solution = {
                        "params": lsq_res.x.copy(),
                        "p_trace": p_trace.copy(),
                        "q_trace": q_trace.copy(),
                        "fourbar_branch": fourbar_branch,
                        "dyad_branch": dyad_branch,
                        "rms_error_bl": rms,
                        "closure_error_bl": float(np.linalg.norm(q_trace[0] - q_trace[-1])),
                        "min_fourbar_upper_margin_bl": float(np.min(diagnostics["fourbar_upper_margin"])),
                        "min_fourbar_lower_margin_bl": float(np.min(diagnostics["fourbar_lower_margin"])),
                        "min_dyad_upper_margin_bl": float(np.min(diagnostics["dyad_upper_margin"])),
                        "min_dyad_lower_margin_bl": float(np.min(diagnostics["dyad_lower_margin"])),
                    }

    if best_solution is None:
        raise RuntimeError("No se encontró una síntesis de seis barras válida.")
    return best_solution


def create_video_sheet(frames: list[np.ndarray]) -> None:
    indices = [0, 20, 40, 60, 80, 100, 120, 160, 200, 240]
    tiles = []
    for frame_idx in indices:
        frame = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 76, 18), fill=(0, 0, 0))
        draw.text((4, 2), f"f={frame_idx}", fill=(255, 255, 255))
        tiles.append(image)

    cols = 2
    rows = math.ceil(len(tiles) / cols)
    canvas = Image.new("RGB", (cols * 320, rows * 240), (255, 255, 255))
    for idx, tile in enumerate(tiles):
        canvas.paste(tile, ((idx % cols) * 320, (idx // cols) * 240))
    canvas.save(FIGURE_SPECS["video_sheet"])


def frame_record_index(distal_records: dict[str, np.ndarray], frame_idx: int) -> int:
    matches = np.where(distal_records["frame"] == frame_idx)[0]
    if len(matches) == 0:
        raise KeyError(f"No hay tracking manual para el frame {frame_idx}.")
    return int(matches[0])


def draw_labeled_marker(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    label: str,
    fill: tuple[int, int, int],
    text_fill: tuple[int, int, int],
    radius: int = 6,
    text_dx: int = 8,
    text_dy: int = -16,
) -> None:
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=fill,
        outline=(0, 0, 0),
        width=2,
    )
    tx = int(round(x + text_dx))
    ty = int(round(y + text_dy))
    tw = 8 + 7 * len(label)
    draw.rounded_rectangle(
        (tx - 3, ty - 2, tx + tw, ty + 12),
        radius=3,
        fill=(0, 0, 0),
    )
    draw.text((tx, ty), label, fill=text_fill)


def create_tracked_points_keyframes(frames: list[np.ndarray], distal_records: dict[str, np.ndarray]) -> None:
    selected_frames = [0, 20, 40, 60, 80, 100]
    tiles = []
    resampling = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    for frame_idx in selected_frames:
        sample_idx = frame_record_index(distal_records, frame_idx)
        frame = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB)
        frame = np.clip(frame.astype(np.float32) * 0.92 + 8.0, 0.0, 255.0).astype(np.uint8)
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        cx = float(distal_records["body_center_x_px"][sample_idx])
        cy = float(distal_records["body_center_y_px"][sample_idx])
        ax = float(distal_records["primary_tip_x_px"][sample_idx])
        ay = float(distal_records["primary_tip_y_px"][sample_idx])
        bx = float(distal_records["secondary_tip_x_px"][sample_idx])
        by = float(distal_records["secondary_tip_y_px"][sample_idx])

        draw.line((cx, cy, ax, ay), fill=(255, 120, 120), width=3)
        draw.line((cx, cy, bx, by), fill=(150, 255, 255), width=3)
        draw_labeled_marker(draw, cx, cy, "cuerpo", (0, 255, 255), (190, 255, 255), radius=7)
        draw_labeled_marker(draw, ax, ay, "A", (255, 80, 80), (255, 240, 180), radius=7)
        draw_labeled_marker(draw, bx, by, "B", (255, 230, 80), (255, 250, 180), radius=7, text_dy=-18)

        all_x = np.array([cx, ax, bx], dtype=float)
        all_y = np.array([cy, ay, by], dtype=float)
        x0 = max(0, int(np.floor(all_x.min() - 58.0)))
        x1 = min(frame.shape[1], int(np.ceil(all_x.max() + 58.0)))
        y0 = max(0, int(np.floor(all_y.min() - 54.0)))
        y1 = min(frame.shape[0], int(np.ceil(all_y.max() + 54.0)))
        crop = image.crop((x0, y0, x1, y1)).resize((320, 240), resample=resampling)
        crop_draw = ImageDraw.Draw(crop)
        crop_draw.rectangle((0, 0, 88, 18), fill=(0, 0, 0))
        crop_draw.text((4, 2), f"f={frame_idx}", fill=(255, 255, 255))
        crop_draw.rounded_rectangle((206, 6, 314, 32), radius=4, fill=(0, 0, 0))
        crop_draw.text((212, 10), "A rojo | B amarillo", fill=(255, 255, 255))
        tiles.append(crop)

    cols = 2
    rows = math.ceil(len(tiles) / cols)
    canvas = Image.new("RGB", (cols * 320, rows * 240), (255, 255, 255))
    for idx, tile in enumerate(tiles):
        canvas.paste(tile, ((idx % cols) * 320, (idx // cols) * 240))
    canvas.save(FIGURE_SPECS["tracked_points_keyframes"])


def create_body_path_overlay(
    frames: list[np.ndarray],
    body_track: dict[str, np.ndarray],
    distal_records: dict[str, np.ndarray],
) -> None:
    resampling = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    background = cv2.cvtColor(frames[50], cv2.COLOR_BGR2RGB)
    background = np.clip(background.astype(np.float32) * 0.60 + 35.0, 0.0, 255.0).astype(np.uint8)

    interval = slice(0, 101)
    body_x = body_track["center_x"][interval]
    body_y = body_track["center_y"][interval]
    primary_x = distal_records["primary_tip_x_px"]
    primary_y = distal_records["primary_tip_y_px"]
    secondary_x = distal_records["secondary_tip_x_px"]
    secondary_y = distal_records["secondary_tip_y_px"]
    all_x = np.concatenate([body_x, primary_x, secondary_x])
    all_y = np.concatenate([body_y, primary_y, secondary_y])
    pad = 28.0
    x0 = max(0, int(np.floor(all_x.min() - pad)))
    x1 = min(background.shape[1], int(np.ceil(all_x.max() + pad)))
    y0 = max(0, int(np.floor(all_y.min() - pad)))
    y1 = min(background.shape[0], int(np.ceil(all_y.max() + pad)))

    image = Image.fromarray(background[y0:y1, x0:x1])
    draw = ImageDraw.Draw(image)

    body_points = list(
        zip(
            (body_x - x0).tolist(),
            (body_y - y0).tolist(),
        )
    )
    if len(body_points) > 1:
        draw.line(body_points, fill=(0, 255, 255), width=4)

    primary_points = list(zip((primary_x - x0).tolist(), (primary_y - y0).tolist()))
    secondary_points = list(zip((secondary_x - x0).tolist(), (secondary_y - y0).tolist()))
    if len(primary_points) > 1:
        draw.line(primary_points, fill=(255, 92, 92), width=4)
    if len(secondary_points) > 1:
        draw.line(secondary_points, fill=(255, 230, 90), width=4)

    for idx, frame_idx in enumerate(distal_records["frame"]):
        ax = float(primary_points[idx][0])
        ay = float(primary_points[idx][1])
        bx = float(secondary_points[idx][0])
        by = float(secondary_points[idx][1])
        draw_labeled_marker(draw, ax, ay, f"A{int(frame_idx)}", (255, 80, 80), (255, 240, 200), radius=5, text_dx=6, text_dy=-14)
        draw_labeled_marker(draw, bx, by, f"B{int(frame_idx)}", (255, 230, 80), (255, 245, 200), radius=5, text_dx=6, text_dy=-14)

    start_point = body_points[0]
    end_point = body_points[-1]
    draw_labeled_marker(draw, start_point[0], start_point[1], "inicio", (255, 255, 255), (255, 255, 255), radius=6)
    draw_labeled_marker(draw, end_point[0], end_point[1], "fin", (0, 255, 255), (220, 255, 255), radius=6)
    draw.rounded_rectangle((8, 8, 180, 40), radius=5, fill=(0, 0, 0))
    draw.text((14, 14), "cian: cuerpo | rojo: A | amarillo: B", fill=(255, 255, 255))

    scale = max(1, int(math.ceil(720 / max(image.width, 1))))
    image = image.resize((image.width * scale, image.height * scale), resample=resampling)
    image.save(FIGURE_SPECS["body_path"])


def create_ciclogram(
    reference_cycle: dict[str, np.ndarray],
    design_cycle: dict[str, np.ndarray | float | int],
    distal_records: dict[str, np.ndarray],
) -> None:
    phase = reference_cycle["phase"]
    xy = reference_cycle["xy"]
    design_xy = np.asarray(design_cycle["xy"], dtype=float)
    control_phase = reference_cycle["control_phase"]
    control_xy = reference_cycle["control_xy"]

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.plot(xy[:, 0], xy[:, 1], color="#0f766e", lw=2.4, label="Referencia periódica del video")
    ax.plot(
        design_xy[:, 0],
        design_xy[:, 1],
        color="#1d4ed8",
        lw=2.0,
        ls="--",
        label=f"Objetivo simplificado (Fourier orden {int(design_cycle['order'])})",
    )
    ax.scatter(
        control_xy[:, 0],
        control_xy[:, 1],
        c=control_phase,
        cmap="plasma",
        s=65,
        edgecolors="black",
        linewidths=0.4,
        label="Puntos rastreados y normalizados",
        zorder=4,
    )

    for frame_idx, x_bl, y_bl in zip(
        distal_records["frame"],
        distal_records["primary_tip_x_bl"],
        distal_records["primary_tip_y_bl"],
    ):
        ax.text(x_bl + 0.008, y_bl + 0.008, f"f={int(frame_idx)}", fontsize=8)

    for phase_name, phase_value in PHASE_MARKERS.items():
        idx = int(np.argmin(np.abs(phase - phase_value)))
        ax.scatter(xy[idx, 0], xy[idx, 1], s=70, marker="x", color="#7c2d12")
        ax.text(xy[idx, 0] + 0.01, xy[idx, 1] + 0.015, phase_name, fontsize=9)

    ax.axhline(0.0, color="#94a3b8", lw=1.0)
    ax.axvline(0.0, color="#94a3b8", lw=1.0)
    ax.text(
        0.02,
        0.03,
        f"RMS simplificación = {float(design_cycle['rms_to_reference_bl']):.4f} BL",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="#cbd5e1"),
    )
    ax.set_title("Ciclograma real y trayectoria objetivo simplificada")
    ax.set_xlabel("x [BL]")
    ax.set_ylabel("y [BL]")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_SPECS["ciclogram"], dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_phase_plot() -> None:
    phase = np.linspace(0.0, 1.0, 400)
    right = is_stance_right(phase).astype(float)
    left = is_stance_left(phase).astype(float)

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.fill_between(
        phase * 100.0,
        1.15,
        1.15 + 0.55 * right,
        step="mid",
        color="#1d4ed8",
        alpha=0.85,
        label="Extremidad A",
    )
    ax.fill_between(
        phase * 100.0,
        0.10,
        0.10 + 0.55 * left,
        step="mid",
        color="#ea580c",
        alpha=0.85,
        label="Extremidad B",
    )
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 2.0)
    ax.set_yticks([0.4, 1.4])
    ax.set_yticklabels(["Extremidad B", "Extremidad A"])
    ax.set_xlabel("Porcentaje del ciclo [%]")
    ax.set_title("Secuencia de fase entre extremidades")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURE_SPECS["phase"], dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_joint_angle_plot(angle_solution: dict[str, np.ndarray]) -> None:
    phase = angle_solution["phase"] * 100.0
    angles_deg = angle_solution["angles_deg"]

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["Pseudo-articulación 1", "Pseudo-articulación 2", "Pseudo-articulación 3"]
    colors = ["#1d4ed8", "#ea580c", "#047857"]
    for idx in range(3):
        ax.plot(phase, angles_deg[:, idx], lw=2.1, color=colors[idx], label=labels[idx])

    for phase_name, phase_value in PHASE_MARKERS.items():
        x = phase_value * 100.0
        ax.axvline(x, color="#475569", ls="--", lw=1.0)
        ax.text(x + 1.0, np.max(angles_deg[:, 0]) - 6.0, phase_name, fontsize=9, color="#334155")

    ax.set_title("Ángulos articulares equivalentes a lo largo del ciclo")
    ax.set_xlabel("Porcentaje del ciclo [%]")
    ax.set_ylabel("Ángulo [deg]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_SPECS["angles"], dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_overlay_plot(
    reference_cycle: dict[str, np.ndarray],
    design_cycle: dict[str, np.ndarray | float | int],
    fourbar_solution: dict[str, np.ndarray | float | int],
) -> None:
    raw_xy = reference_cycle["xy"]
    design_xy = np.asarray(design_cycle["xy"], dtype=float)
    trace = np.asarray(fourbar_solution["trace"], dtype=float)
    phase = reference_cycle["phase"]

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.plot(raw_xy[:, 0], raw_xy[:, 1], color="#0f766e", lw=2.4, label="Trayectoria real del video")
    ax.plot(design_xy[:, 0], design_xy[:, 1], color="#1d4ed8", lw=2.0, ls="--", label="Objetivo simplificado")
    ax.plot(trace[:, 0], trace[:, 1], color="#dc2626", lw=2.0, ls="-.", label="Trayectoria ejecutada por 4 barras")

    for phase_name, phase_value in PHASE_MARKERS.items():
        idx = int(np.argmin(np.abs(phase - phase_value)))
        ax.scatter(raw_xy[idx, 0], raw_xy[idx, 1], color="#0f766e", s=40)
        ax.scatter(design_xy[idx, 0], design_xy[idx, 1], color="#1d4ed8", s=35)
        ax.scatter(trace[idx, 0], trace[idx, 1], color="#dc2626", s=40)

    ax.set_title("4 barras: trayectoria real vs objetivo simplificado vs ejecutada")
    ax.set_xlabel("x [BL]")
    ax.set_ylabel("y [BL]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_SPECS["overlay"], dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_mechanism_snapshots(fourbar_solution: dict[str, np.ndarray | float | int], reference_cycle: dict[str, np.ndarray]) -> None:
    params = np.asarray(fourbar_solution["params"], dtype=float)
    branch = int(fourbar_solution["branch"])
    snapshot_phases = [PHASE_MARKERS["contacto"], PHASE_MARKERS["apoyo"], PHASE_MARKERS["despegue"], PHASE_MARKERS["vuelo"]]
    all_points = [reference_cycle["xy"]]
    states = []
    for phase_value in snapshot_phases:
        state = fourbar_state(params, float(phase_value), branch=branch)
        if state is None:
            continue
        states.append((phase_value, state))
        all_points.append(
            np.vstack(
                [
                    np.asarray(state["o2"], dtype=float),
                    np.asarray(state["a"], dtype=float),
                    np.asarray(state["b"], dtype=float),
                    np.asarray(state["o4"], dtype=float),
                    np.asarray(state["p"], dtype=float),
                ]
            )
        )

    stacked = np.vstack(all_points)
    pad = 0.08
    x_limits = (float(np.min(stacked[:, 0]) - pad), float(np.max(stacked[:, 0]) + pad))
    y_limits = (float(np.min(stacked[:, 1]) - pad), float(np.max(stacked[:, 1]) + pad))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for ax, (phase_value, state) in zip(axes, states):
        o2 = np.asarray(state["o2"], dtype=float)
        a_point = np.asarray(state["a"], dtype=float)
        b_point = np.asarray(state["b"], dtype=float)
        o4 = np.asarray(state["o4"], dtype=float)
        p_point = np.asarray(state["p"], dtype=float)
        coupler_plate = plt.Polygon(
            [a_point, b_point, p_point],
            closed=True,
            facecolor="#c4b5fd",
            edgecolor="none",
            alpha=0.28,
            zorder=1,
        )
        ax.add_patch(coupler_plate)

        ax.plot([o2[0], o4[0]], [o2[1], o4[1]], color="#111827", lw=2.2, ls=":", label="Base fija")
        ax.plot([o2[0], a_point[0]], [o2[1], a_point[1]], color="#1d4ed8", lw=2.6, label="Entrada")
        ax.plot([a_point[0], b_point[0]], [a_point[1], b_point[1]], color="#dc2626", lw=2.6, label="Acoplador")
        ax.plot([b_point[0], o4[0]], [b_point[1], o4[1]], color="#059669", lw=2.6, label="Salida")
        ax.plot([a_point[0], p_point[0]], [a_point[1], p_point[1]], color="#7c3aed", lw=2.2, ls="--", label="Offset rígido hacia P")
        ax.plot([b_point[0], p_point[0]], [b_point[1], p_point[1]], color="#7c3aed", lw=1.8, ls="--")
        ax.plot(reference_cycle["xy"][:, 0], reference_cycle["xy"][:, 1], color="#94a3b8", alpha=0.5)
        ax.scatter(
            [o2[0], o4[0], a_point[0], b_point[0], p_point[0]],
            [o2[1], o4[1], a_point[1], b_point[1], p_point[1]],
            color="#111827",
            s=24,
            zorder=4,
        )
        ax.text(o2[0] + 0.01, o2[1] + 0.015, "O2", fontsize=8, color="#111827")
        ax.text(o4[0] + 0.01, o4[1] + 0.015, "O4", fontsize=8, color="#111827")
        ax.text(a_point[0] + 0.01, a_point[1] + 0.015, "A", fontsize=8, color="#1d4ed8")
        ax.text(b_point[0] + 0.01, b_point[1] + 0.015, "B", fontsize=8, color="#dc2626")
        ax.text(p_point[0] + 0.01, p_point[1] + 0.015, "P", fontsize=8, color="#7c3aed")
        ax.set_title(f"Fase = {phase_value * 100:.0f}%")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Estados clave del mismo mecanismo de 4 barras", y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.11)
    fig.savefig(FIGURE_SPECS["mechanism"], dpi=180, bbox_inches="tight")
    plt.close(fig)


def segment_intersection(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> bool:
    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        return (
            min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9
            and min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9
        )

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)

    if (o1 * o2 < 0.0) and (o3 * o4 < 0.0):
        return True
    if abs(o1) < 1e-9 and on_segment(p1, q1, p2):
        return True
    if abs(o2) < 1e-9 and on_segment(p1, q2, p2):
        return True
    if abs(o3) < 1e-9 and on_segment(q1, p1, q2):
        return True
    if abs(o4) < 1e-9 and on_segment(q1, p2, q2):
        return True
    return False


def point_to_segment_distance(point: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    segment = seg_b - seg_a
    denom = float(np.dot(segment, segment))
    if denom < 1e-12:
        return float(np.linalg.norm(point - seg_a))
    t = float(np.dot(point - seg_a, segment) / denom)
    t = min(1.0, max(0.0, t))
    projection = seg_a + t * segment
    return float(np.linalg.norm(point - projection))


def segment_distance(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> float:
    if segment_intersection(p1, p2, q1, q2):
        return 0.0
    return min(
        point_to_segment_distance(p1, q1, q2),
        point_to_segment_distance(p2, q1, q2),
        point_to_segment_distance(q1, p1, p2),
        point_to_segment_distance(q2, p1, p2),
    )


def fourbar_link_segments(state: dict[str, np.ndarray | float]) -> list[dict[str, object]]:
    o2 = np.asarray(state["o2"], dtype=float)
    a_point = np.asarray(state["a"], dtype=float)
    b_point = np.asarray(state["b"], dtype=float)
    o4 = np.asarray(state["o4"], dtype=float)
    p_point = np.asarray(state["p"], dtype=float)
    return [
        {"name": "ground_O2O4", "p1": o2, "p2": o4, "joints": {"O2", "O4"}, "layer": "lower"},
        {"name": "input_O2A", "p1": o2, "p2": a_point, "joints": {"O2", "A"}, "layer": "upper"},
        {"name": "coupler_AB", "p1": a_point, "p2": b_point, "joints": {"A", "B"}, "layer": "upper"},
        {"name": "output_BO4", "p1": b_point, "p2": o4, "joints": {"B", "O4"}, "layer": "lower"},
        {"name": "plate_AP", "p1": a_point, "p2": p_point, "joints": {"A", "P"}, "layer": "upper"},
        {"name": "plate_BP", "p1": b_point, "p2": p_point, "joints": {"B", "P"}, "layer": "upper"},
    ]


def fourbar_collision_metrics(
    state: dict[str, np.ndarray | float],
    phase_value: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    segments = fourbar_link_segments(state)
    rows: list[dict[str, object]] = []
    min_clearance_projected = float("inf")
    worst_pair_projected = ""
    min_clearance_physical = float("inf")
    worst_pair_physical = ""
    any_projected_collision = False
    any_physical_collision = False

    for idx in range(len(segments)):
        for jdx in range(idx + 1, len(segments)):
            first = segments[idx]
            second = segments[jdx]
            if first["joints"] & second["joints"]:
                continue
            p1 = np.asarray(first["p1"], dtype=float)
            p2 = np.asarray(first["p2"], dtype=float)
            q1 = np.asarray(second["p1"], dtype=float)
            q2 = np.asarray(second["p2"], dtype=float)
            clearance = segment_distance(p1, p2, q1, q2)
            projected_collision = clearance < 1e-6
            same_layer = str(first["layer"]) == str(second["layer"])
            physical_collision = projected_collision and same_layer
            any_projected_collision = any_projected_collision or projected_collision
            any_physical_collision = any_physical_collision or physical_collision
            if clearance < min_clearance_projected:
                min_clearance_projected = float(clearance)
                worst_pair_projected = f"{first['name']} vs {second['name']}"
            if same_layer and clearance < min_clearance_physical:
                min_clearance_physical = float(clearance)
                worst_pair_physical = f"{first['name']} vs {second['name']}"
            rows.append(
                {
                    "phase_fraction": float(phase_value),
                    "segment_a": str(first["name"]),
                    "segment_b": str(second["name"]),
                    "layer_a": str(first["layer"]),
                    "layer_b": str(second["layer"]),
                    "same_layer": int(same_layer),
                    "clearance_bl": float(clearance),
                    "projected_collision": int(projected_collision),
                    "physical_collision": int(physical_collision),
                }
            )

    if not math.isfinite(min_clearance_projected):
        min_clearance_projected = 1.0
    if not math.isfinite(min_clearance_physical):
        min_clearance_physical = 1.0

    summary = {
        "min_clearance_projected_bl": float(min_clearance_projected),
        "worst_pair_projected": worst_pair_projected,
        "min_clearance_physical_bl": float(min_clearance_physical),
        "worst_pair_physical": worst_pair_physical,
        "any_projected_collision": bool(any_projected_collision),
        "any_physical_collision": bool(any_physical_collision),
    }
    return rows, summary


def evaluate_fourbar_collisions(
    fourbar_solution: dict[str, np.ndarray | float | int],
    n_samples: int = 361,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    params = np.asarray(fourbar_solution["params"], dtype=float)
    branch = int(fourbar_solution["branch"])
    phases = np.linspace(0.0, 1.0, n_samples)
    rows: list[dict[str, object]] = []
    min_clearance_projected = float("inf")
    worst_pair_projected = ""
    worst_phase_projected = 0.0
    min_clearance_physical = float("inf")
    worst_pair_physical = ""
    worst_phase_physical = 0.0
    any_projected_collision = False
    any_physical_collision = False
    for phase_value in phases:
        state = fourbar_state(params, float(phase_value), branch=branch)
        if state is None:
            continue
        phase_rows, phase_summary = fourbar_collision_metrics(state, float(phase_value))
        rows.extend(phase_rows)
        any_projected_collision = any_projected_collision or bool(phase_summary["any_projected_collision"])
        any_physical_collision = any_physical_collision or bool(phase_summary["any_physical_collision"])
        if float(phase_summary["min_clearance_projected_bl"]) < min_clearance_projected:
            min_clearance_projected = float(phase_summary["min_clearance_projected_bl"])
            worst_pair_projected = str(phase_summary["worst_pair_projected"])
            worst_phase_projected = float(phase_value)
        if float(phase_summary["min_clearance_physical_bl"]) < min_clearance_physical:
            min_clearance_physical = float(phase_summary["min_clearance_physical_bl"])
            worst_pair_physical = str(phase_summary["worst_pair_physical"])
            worst_phase_physical = float(phase_value)

    summary = {
        "min_clearance_projected_bl": float(min_clearance_projected),
        "worst_pair_projected": worst_pair_projected,
        "worst_phase_projected_fraction": float(worst_phase_projected),
        "any_projected_collision": bool(any_projected_collision),
        "min_clearance_physical_bl": float(min_clearance_physical if math.isfinite(min_clearance_physical) else 1.0),
        "worst_pair_physical": worst_pair_physical,
        "worst_phase_physical_fraction": float(worst_phase_physical),
        "any_physical_collision": bool(any_physical_collision),
        "max_recommended_link_radius_bl": 0.5
        * float(min_clearance_physical if math.isfinite(min_clearance_physical) else 1.0),
        "layering_note": "La capa lower contiene suelo y salida; la capa upper contiene entrada y placa del acoplador.",
    }
    return rows, summary


def create_fourbar_full_cycle_plot(
    reference_cycle: dict[str, np.ndarray],
    design_cycle: dict[str, np.ndarray | float | int],
    fourbar_solution: dict[str, np.ndarray | float | int],
    collision_summary: dict[str, object],
) -> None:
    params = np.asarray(fourbar_solution["params"], dtype=float)
    branch = int(fourbar_solution["branch"])
    phase_dense = np.linspace(0.0, 1.0, 240)
    a_path = []
    b_path = []
    p_path = []
    poses = []
    for phase_value in phase_dense:
        state = fourbar_state(params, float(phase_value), branch=branch)
        if state is None:
            continue
        a_path.append(np.asarray(state["a"], dtype=float))
        b_path.append(np.asarray(state["b"], dtype=float))
        p_path.append(np.asarray(state["p"], dtype=float))
    for phase_value in np.linspace(0.0, 1.0, 12, endpoint=False):
        state = fourbar_state(params, float(phase_value), branch=branch)
        if state is not None:
            poses.append(state)

    a_path = np.asarray(a_path, dtype=float)
    b_path = np.asarray(b_path, dtype=float)
    p_path = np.asarray(p_path, dtype=float)
    raw_xy = np.asarray(reference_cycle["xy"], dtype=float)
    design_xy = np.asarray(design_cycle["xy"], dtype=float)

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.plot(raw_xy[:, 0], raw_xy[:, 1], color="#0f766e", lw=2.2, label="Trayectoria real del video")
    ax.plot(design_xy[:, 0], design_xy[:, 1], color="#1d4ed8", lw=2.0, ls="--", label="Objetivo simplificado")
    ax.plot(a_path[:, 0], a_path[:, 1], color="#2563eb", lw=1.6, alpha=0.8, label="Trayectoria de A")
    ax.plot(b_path[:, 0], b_path[:, 1], color="#059669", lw=1.6, alpha=0.8, label="Trayectoria de B")
    ax.plot(p_path[:, 0], p_path[:, 1], color="#dc2626", lw=2.2, label="Trayectoria de P (4 barras)")

    for state in poses:
        o2 = np.asarray(state["o2"], dtype=float)
        a_point = np.asarray(state["a"], dtype=float)
        b_point = np.asarray(state["b"], dtype=float)
        o4 = np.asarray(state["o4"], dtype=float)
        p_point = np.asarray(state["p"], dtype=float)
        ax.plot([o2[0], a_point[0]], [o2[1], a_point[1]], color="#2563eb", lw=0.8, alpha=0.18)
        ax.plot([a_point[0], b_point[0]], [a_point[1], b_point[1]], color="#dc2626", lw=0.8, alpha=0.18)
        ax.plot([b_point[0], o4[0]], [b_point[1], o4[1]], color="#059669", lw=0.8, alpha=0.18)
        ax.plot([a_point[0], p_point[0]], [a_point[1], p_point[1]], color="#7c3aed", lw=0.7, ls="--", alpha=0.16)
        ax.plot([b_point[0], p_point[0]], [b_point[1], p_point[1]], color="#7c3aed", lw=0.7, ls="--", alpha=0.16)

    ax.text(
        0.02,
        0.03,
        "Choque físico por capas: "
        f"{'sí' if bool(collision_summary['any_physical_collision']) else 'no'}"
        f" | clearance mínimo misma capa = {float(collision_summary['min_clearance_physical_bl']):.4f} BL",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="#cbd5e1"),
    )
    ax.set_title("Trayectoria completa del 4 barras base")
    ax.set_xlabel("x [BL]")
    ax.set_ylabel("y [BL]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_SPECS["fourbar_full_cycle"], dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_fourbar_state_rows(
    fourbar_solution: dict[str, np.ndarray | float | int],
    phase: np.ndarray,
) -> list[dict[str, object]]:
    params = np.asarray(fourbar_solution["params"], dtype=float)
    branch = int(fourbar_solution["branch"])
    rows: list[dict[str, object]] = []
    for phase_value in phase:
        state = fourbar_state(params, float(phase_value), branch=branch)
        if state is None:
            continue
        _, collision_summary = fourbar_collision_metrics(state, float(phase_value))
        o2 = np.asarray(state["o2"], dtype=float)
        a_point = np.asarray(state["a"], dtype=float)
        b_point = np.asarray(state["b"], dtype=float)
        o4 = np.asarray(state["o4"], dtype=float)
        p_point = np.asarray(state["p"], dtype=float)
        theta4 = math.atan2(b_point[1] - o4[1], b_point[0] - o4[0])
        rows.append(
            {
                "phase_fraction": float(phase_value),
                "O2_x_bl": float(o2[0]),
                "O2_y_bl": float(o2[1]),
                "A_x_bl": float(a_point[0]),
                "A_y_bl": float(a_point[1]),
                "B_x_bl": float(b_point[0]),
                "B_y_bl": float(b_point[1]),
                "O4_x_bl": float(o4[0]),
                "O4_y_bl": float(o4[1]),
                "P_x_bl": float(p_point[0]),
                "P_y_bl": float(p_point[1]),
                "theta2_deg": float(np.rad2deg(float(state["theta2"]))),
                "theta3_deg": float(np.rad2deg(float(state["theta3"]))),
                "theta4_deg": float(np.rad2deg(theta4)),
                "upper_margin_bl": float(state["upper_margin"]),
                "lower_margin_bl": float(state["lower_margin"]),
                "min_projected_clearance_bl": float(collision_summary["min_clearance_projected_bl"]),
                "min_same_layer_clearance_bl": float(collision_summary["min_clearance_physical_bl"]),
                "projected_collision": int(bool(collision_summary["any_projected_collision"])),
                "physical_collision": int(bool(collision_summary["any_physical_collision"])),
            }
        )
    return rows


def create_fourbar_clearance_plot(fourbar_state_rows: list[dict[str, object]]) -> None:
    phase_pct = np.array([100.0 * float(row["phase_fraction"]) for row in fourbar_state_rows], dtype=float)
    min_projected = np.array([float(row["min_projected_clearance_bl"]) for row in fourbar_state_rows], dtype=float)
    min_same_layer = np.array([float(row["min_same_layer_clearance_bl"]) for row in fourbar_state_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(phase_pct, min_projected, color="#dc2626", lw=2.0, label="Clearance proyectado 2D")
    ax.plot(phase_pct, min_same_layer, color="#059669", lw=2.3, label="Clearance misma capa")
    ax.axhline(0.0, color="#111827", lw=1.0, ls=":")
    ax.fill_between(phase_pct, 0.0, min_same_layer, color="#86efac", alpha=0.18)
    ax.set_title("Clearance mínimo del 4 barras a lo largo del ciclo")
    ax.set_xlabel("Porcentaje del ciclo [%]")
    ax.set_ylabel("Clearance [BL]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_SPECS["fourbar_clearance"], dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_fourbar_cable_tpu_concept_plot(
    fourbar_solution: dict[str, np.ndarray | float | int],
) -> None:
    params = np.asarray(fourbar_solution["params"], dtype=float)
    branch = int(fourbar_solution["branch"])
    candidate_phases = np.linspace(0.0, 1.0, 121)
    best_phase = None
    best_score = -np.inf
    for phase_value in candidate_phases:
        candidate_state = fourbar_state(params, float(phase_value), branch=branch)
        if candidate_state is None:
            continue
        _, candidate_summary = fourbar_collision_metrics(candidate_state, float(phase_value))
        if bool(candidate_summary["any_projected_collision"]):
            continue
        score = float(candidate_summary["min_clearance_physical_bl"])
        if score > best_score:
            best_score = score
            best_phase = float(phase_value)

    clean_phase = float(best_phase) if best_phase is not None else float(PHASE_MARKERS["apoyo"])
    state = fourbar_state(params, clean_phase, branch=branch)
    if state is None:
        state = fourbar_state(params, float(PHASE_MARKERS["apoyo"]), branch=branch)
    if state is None:
        raise RuntimeError("No fue posible construir el esquema 4 barras + cable.")
    o2 = np.asarray(state["o2"], dtype=float)
    a_point = np.asarray(state["a"], dtype=float)
    b_point = np.asarray(state["b"], dtype=float)
    o4 = np.asarray(state["o4"], dtype=float)
    p_point = np.asarray(state["p"], dtype=float)
    mechanism_points = np.vstack([o2, a_point, b_point, o4, p_point])
    max_x = float(np.max(mechanism_points[:, 0]))
    min_y = float(np.min(mechanism_points[:, 1]))
    max_y = float(np.max(mechanism_points[:, 1]))
    pend_pivot = np.array([max_x + 0.14, 0.5 * (min_y + max_y) - 0.03], dtype=float)
    tentacle_tip = pend_pivot + np.array([0.06, -0.38], dtype=float)
    cable_attach = pend_pivot + 0.58 * (tentacle_tip - pend_pivot)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    coupler_plate = plt.Polygon(
        [a_point, b_point, p_point],
        closed=True,
        facecolor="#c4b5fd",
        edgecolor="#7c3aed",
        alpha=0.22,
        zorder=1,
    )
    ax.add_patch(coupler_plate)
    ax.plot([o2[0], o4[0]], [o2[1], o4[1]], color="#111827", lw=2.6, ls=":", label="Base fija O2-O4")
    ax.plot([o2[0], a_point[0]], [o2[1], a_point[1]], color="#2563eb", lw=3.0, label="Entrada O2-A")
    ax.plot([a_point[0], b_point[0]], [a_point[1], b_point[1]], color="#dc2626", lw=3.0, label="Acoplador AB")
    ax.plot([b_point[0], o4[0]], [b_point[1], o4[1]], color="#059669", lw=3.0, label="Salida B-O4")
    ax.plot([a_point[0], p_point[0]], [a_point[1], p_point[1]], color="#7c3aed", lw=2.0, ls="--", label="Placa rígida A-B-P")
    ax.plot([b_point[0], p_point[0]], [b_point[1], p_point[1]], color="#7c3aed", lw=2.0, ls="--")
    ax.plot([p_point[0], cable_attach[0]], [p_point[1], cable_attach[1]], color="#f59e0b", lw=2.2, ls="-.", label="Cable de tensión")

    control_points = np.array(
        [
            pend_pivot,
            pend_pivot + np.array([0.05, -0.06]),
            pend_pivot + np.array([0.09, -0.17]),
            pend_pivot + np.array([0.10, -0.30]),
            tentacle_tip,
        ],
        dtype=float,
    )
    ax.plot(control_points[:, 0], control_points[:, 1], color="#84cc16", lw=7.0, solid_capstyle="round", label="Tentáculo TPU")
    ax.plot(control_points[:, 0], control_points[:, 1], color="#365314", lw=1.3)
    ax.plot([pend_pivot[0], tentacle_tip[0]], [pend_pivot[1], tentacle_tip[1]], color="#64748b", lw=1.6, alpha=0.7, label="Miembro tipo péndulo")

    ax.scatter(
        [o2[0], o4[0], a_point[0], b_point[0], p_point[0], pend_pivot[0]],
        [o2[1], o4[1], a_point[1], b_point[1], p_point[1], pend_pivot[1]],
        color="#111827",
        s=36,
        zorder=5,
    )
    for label, point, color in [
        ("O2", o2, "#111827"),
        ("O4", o4, "#111827"),
        ("A", a_point, "#2563eb"),
        ("B", b_point, "#dc2626"),
        ("P", p_point, "#7c3aed"),
        ("Ot", pend_pivot, "#111827"),
    ]:
        ax.text(point[0] + 0.01, point[1] + 0.015, label, fontsize=10, color=color)
    ax.text(cable_attach[0] + 0.01, cable_attach[1], "anclaje cable", fontsize=9, color="#b45309")
    ax.text(tentacle_tip[0] + 0.01, tentacle_tip[1] - 0.01, "punta TPU", fontsize=9, color="#4d7c0f")
    ax.text(
        0.02,
        0.03,
        f"Fase mostrada = {clean_phase * 100:.1f}% | configuración elegida sin cruce proyectado",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="#cbd5e1"),
    )
    ax.set_title("Concepto final: 4 barras + cable tensando tentáculo TPU")
    ax.set_xlabel("x [BL]")
    ax.set_ylabel("y [BL]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_SPECS["fourbar_cable_concept"], dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_sixbar_overlay_plot(
    reference_cycle: dict[str, np.ndarray],
    design_cycle: dict[str, np.ndarray | float | int],
    fourbar_solution: dict[str, np.ndarray | float | int],
    sixbar_solution: dict[str, np.ndarray | float | int],
) -> None:
    raw_xy = reference_cycle["xy"]
    design_xy = np.asarray(design_cycle["xy"], dtype=float)
    p_trace = np.asarray(sixbar_solution["p_trace"], dtype=float)
    q_trace = np.asarray(sixbar_solution["q_trace"], dtype=float)
    phase = reference_cycle["phase"]

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.plot(raw_xy[:, 0], raw_xy[:, 1], color="#0f766e", lw=2.6, label="Trayectoria real del video")
    ax.plot(design_xy[:, 0], design_xy[:, 1], color="#1d4ed8", lw=2.0, ls="--", label="Objetivo simplificado")
    ax.plot(
        np.asarray(fourbar_solution["trace"], dtype=float)[:, 0],
        np.asarray(fourbar_solution["trace"], dtype=float)[:, 1],
        color="#dc2626",
        lw=1.8,
        ls="--",
        label="P del 4 barras base",
    )
    ax.plot(q_trace[:, 0], q_trace[:, 1], color="#ea580c", lw=2.2, label="Q ejecutada por 6 barras")
    for phase_name, phase_value in PHASE_MARKERS.items():
        idx = int(np.argmin(np.abs(phase - phase_value)))
        ax.scatter(raw_xy[idx, 0], raw_xy[idx, 1], color="#0f766e", s=38)
        ax.scatter(design_xy[idx, 0], design_xy[idx, 1], color="#1d4ed8", s=34)
        ax.scatter(p_trace[idx, 0], p_trace[idx, 1], color="#7c3aed", s=36)
        ax.scatter(q_trace[idx, 0], q_trace[idx, 1], color="#ea580c", s=38)
        ax.text(q_trace[idx, 0] + 0.012, q_trace[idx, 1] + 0.012, phase_name[:3], fontsize=8, color="#9a3412")

    ax.set_title("6 barras: trayectoria real vs objetivo simplificado vs ejecutada")
    ax.set_xlabel("x [BL]")
    ax.set_ylabel("y [BL]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_SPECS["sixbar_overlay"], dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_sixbar_snapshots(
    sixbar_solution: dict[str, np.ndarray | float | int],
    reference_cycle: dict[str, np.ndarray],
) -> None:
    params = np.asarray(sixbar_solution["params"], dtype=float)
    fourbar_branch = int(sixbar_solution["fourbar_branch"])
    dyad_branch = int(sixbar_solution["dyad_branch"])
    snapshot_phases = [
        PHASE_MARKERS["contacto"],
        PHASE_MARKERS["apoyo"],
        PHASE_MARKERS["despegue"],
        PHASE_MARKERS["vuelo"],
    ]
    states = []
    all_points = [reference_cycle["xy"], np.asarray(sixbar_solution["q_trace"], dtype=float)]
    for phase_value in snapshot_phases:
        state = sixbar_state(params, float(phase_value), fourbar_branch=fourbar_branch, dyad_branch=dyad_branch)
        if state is None:
            continue
        states.append((phase_value, state))
        all_points.append(
            np.vstack(
                [
                    np.asarray(state["o2"], dtype=float),
                    np.asarray(state["a"], dtype=float),
                    np.asarray(state["b"], dtype=float),
                    np.asarray(state["o4"], dtype=float),
                    np.asarray(state["p"], dtype=float),
                    np.asarray(state["o6"], dtype=float),
                    np.asarray(state["q"], dtype=float),
                ]
            )
        )

    stacked = np.vstack(all_points)
    pad = 0.08
    x_limits = (float(np.min(stacked[:, 0]) - pad), float(np.max(stacked[:, 0]) + pad))
    y_limits = (float(np.min(stacked[:, 1]) - pad), float(np.max(stacked[:, 1]) + pad))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
    axes = axes.ravel()
    q_trace = np.asarray(sixbar_solution["q_trace"], dtype=float)
    for ax, (phase_value, state) in zip(axes, states):
        o2 = np.asarray(state["o2"], dtype=float)
        a_point = np.asarray(state["a"], dtype=float)
        b_point = np.asarray(state["b"], dtype=float)
        o4 = np.asarray(state["o4"], dtype=float)
        p_point = np.asarray(state["p"], dtype=float)
        o6 = np.asarray(state["o6"], dtype=float)
        q_point = np.asarray(state["q"], dtype=float)
        pq_len = float(state["pq_len"])
        o6q_len = float(state["o6q_len"])

        coupler_plate = plt.Polygon(
            [a_point, b_point, p_point],
            closed=True,
            facecolor="#c4b5fd",
            edgecolor="#7c3aed",
            alpha=0.22,
            zorder=1,
        )
        ax.add_patch(coupler_plate)
        ax.add_patch(plt.Circle(p_point, pq_len, edgecolor="#f59e0b", facecolor="none", lw=1.0, ls="--", alpha=0.35))
        ax.add_patch(plt.Circle(o6, o6q_len, edgecolor="#92400e", facecolor="none", lw=1.0, ls="--", alpha=0.35))

        ax.plot([o2[0], o4[0]], [o2[1], o4[1]], color="#111827", lw=2.2, ls=":", label="Base fija O2-O4")
        ax.plot([o2[0], a_point[0]], [o2[1], a_point[1]], color="#2563eb", lw=2.6, label="Entrada O2-A")
        ax.plot([a_point[0], b_point[0]], [a_point[1], b_point[1]], color="#dc2626", lw=2.6, label="Acoplador AB")
        ax.plot([b_point[0], o4[0]], [b_point[1], o4[1]], color="#059669", lw=2.6, label="Salida B-O4")
        ax.plot([a_point[0], p_point[0]], [a_point[1], p_point[1]], color="#7c3aed", lw=2.0, ls="--", label="Placa A-B-P")
        ax.plot([b_point[0], p_point[0]], [b_point[1], p_point[1]], color="#7c3aed", lw=2.0, ls="--")
        ax.plot([p_point[0], q_point[0]], [p_point[1], q_point[1]], color="#ea580c", lw=2.6, label="Eslabón P-Q")
        ax.plot([o6[0], q_point[0]], [o6[1], q_point[1]], color="#92400e", lw=2.6, label="Eslabón O6-Q")
        ax.plot(reference_cycle["xy"][:, 0], reference_cycle["xy"][:, 1], color="#94a3b8", alpha=0.5, lw=1.5)
        ax.plot(q_trace[:, 0], q_trace[:, 1], color="#ea580c", alpha=0.4, lw=1.6)
        ax.scatter(
            [o2[0], o4[0], o6[0], a_point[0], b_point[0], p_point[0], q_point[0]],
            [o2[1], o4[1], o6[1], a_point[1], b_point[1], p_point[1], q_point[1]],
            color="#111827",
            s=24,
            zorder=4,
        )
        ax.text(o2[0] + 0.01, o2[1] + 0.015, "O2", fontsize=8)
        ax.text(o4[0] + 0.01, o4[1] + 0.015, "O4", fontsize=8)
        ax.text(o6[0] + 0.01, o6[1] + 0.015, "O6", fontsize=8)
        ax.text(a_point[0] + 0.01, a_point[1] + 0.015, "A", fontsize=8, color="#1d4ed8")
        ax.text(b_point[0] + 0.01, b_point[1] + 0.015, "B", fontsize=8, color="#dc2626")
        ax.text(p_point[0] + 0.01, p_point[1] + 0.015, "P", fontsize=8, color="#7c3aed")
        ax.text(q_point[0] + 0.01, q_point[1] + 0.015, "Q", fontsize=8, color="#ea580c")
        ax.set_title(f"Fase = {phase_value * 100:.0f}%")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Estados clave del mecanismo 6 barras sintetizado", y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(FIGURE_SPECS["sixbar_snapshots"], dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_single_sixbar_module_plot(sixbar_solution: dict[str, np.ndarray | float | int]) -> None:
    params = np.asarray(sixbar_solution["params"], dtype=float)
    fourbar_branch = int(sixbar_solution["fourbar_branch"])
    dyad_branch = int(sixbar_solution["dyad_branch"])
    state = sixbar_state(
        params,
        float(PHASE_MARKERS["apoyo"]),
        fourbar_branch=fourbar_branch,
        dyad_branch=dyad_branch,
    )
    if state is None:
        raise RuntimeError("No fue posible construir la figura del módulo 6 barras.")

    o2 = np.asarray(state["o2"], dtype=float)
    a_point = np.asarray(state["a"], dtype=float)
    b_point = np.asarray(state["b"], dtype=float)
    o4 = np.asarray(state["o4"], dtype=float)
    p_point = np.asarray(state["p"], dtype=float)
    o6 = np.asarray(state["o6"], dtype=float)
    q_point = np.asarray(state["q"], dtype=float)
    pq_len = float(state["pq_len"])
    o6q_len = float(state["o6q_len"])
    all_points = np.vstack([o2, a_point, b_point, o4, p_point, o6, q_point])
    pad = 0.22
    x_limits = (float(np.min(all_points[:, 0]) - pad), float(np.max(all_points[:, 0]) + pad))
    y_limits = (float(np.min(all_points[:, 1]) - pad), float(np.max(all_points[:, 1]) + pad))

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.16)

    coupler_plate = plt.Polygon(
        [a_point, b_point, p_point],
        closed=True,
        facecolor="#c4b5fd",
        edgecolor="#7c3aed",
        alpha=0.24,
        zorder=1,
    )
    ax.add_patch(coupler_plate)
    ax.add_patch(plt.Circle(p_point, pq_len, edgecolor="#f59e0b", facecolor="none", lw=1.3, ls="--", alpha=0.45))
    ax.add_patch(plt.Circle(o6, o6q_len, edgecolor="#92400e", facecolor="none", lw=1.3, ls="--", alpha=0.45))

    ax.plot([o2[0], o4[0]], [o2[1], o4[1]], color="#111827", lw=2.6, ls=":", label="Base fija O2-O4")
    ax.plot([o2[0], a_point[0]], [o2[1], a_point[1]], color="#2563eb", lw=3.0, label="Entrada O2-A")
    ax.plot([a_point[0], b_point[0]], [a_point[1], b_point[1]], color="#dc2626", lw=3.0, label="Acoplador AB")
    ax.plot([b_point[0], o4[0]], [b_point[1], o4[1]], color="#059669", lw=3.0, label="Salida B-O4")
    ax.plot([a_point[0], p_point[0]], [a_point[1], p_point[1]], color="#7c3aed", lw=2.2, ls="--", label="Placa rígida A-B-P")
    ax.plot([b_point[0], p_point[0]], [b_point[1], p_point[1]], color="#7c3aed", lw=2.2, ls="--")
    ax.plot([p_point[0], q_point[0]], [p_point[1], q_point[1]], color="#ea580c", lw=3.0, label="Eslabón P-Q")
    ax.plot([o6[0], q_point[0]], [o6[1], q_point[1]], color="#92400e", lw=3.0, label="Eslabón O6-Q")
    ax.scatter(
        [o2[0], o4[0], o6[0], a_point[0], b_point[0], p_point[0], q_point[0]],
        [o2[1], o4[1], o6[1], a_point[1], b_point[1], p_point[1], q_point[1]],
        s=52,
        color="#111827",
        zorder=5,
    )
    for label, point, color in [
        ("O2", o2, "#111827"),
        ("O4", o4, "#111827"),
        ("O6", o6, "#111827"),
        ("A", a_point, "#2563eb"),
        ("B", b_point, "#dc2626"),
        ("P", p_point, "#7c3aed"),
        ("Q", q_point, "#ea580c"),
    ]:
        ax.text(point[0] + 0.012, point[1] + 0.015, label, fontsize=10, color=color)

    ax.annotate("", xy=o2 + np.array([0.09, -0.09]), xytext=o2 + np.array([-0.07, 0.17]), arrowprops=dict(arrowstyle="->", color="#2563eb", lw=2.0))
    ax.text(o2[0] - 0.02, o2[1] + 0.2, r"$\omega$", fontsize=12, color="#2563eb")
    ax.text(
        float(x_limits[0] + 0.02),
        float(y_limits[1] - 0.06),
        "Q = intersección de los círculos centrados en P y O6",
        fontsize=10.5,
        color="#334155",
        ha="left",
        va="top",
    )
    ax.text(
        float(x_limits[0] + 0.02),
        float(y_limits[1] - 0.13),
        "Si faltara uno de esos dos eslabones, Q sí quedaría libre.",
        fontsize=10.0,
        color="#475569",
        ha="left",
        va="top",
    )
    ax.set_title("Módulo 6 barras por brazo")
    ax.set_xlabel("x [BL]")
    ax.set_ylabel("y [BL]")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURE_SPECS["sixbar_module"], dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_dual_architecture_plot() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 5.2))
    ax.set_xlim(-6.4, 6.4)
    ax.set_ylim(-2.85, 2.9)
    ax.axis("off")

    left_plate = plt.Rectangle((-5.55, -1.42), 3.6, 2.8, facecolor="#e2e8f0", edgecolor="#94a3b8", lw=1.8, alpha=0.32)
    right_plate = plt.Rectangle((1.95, -1.42), 3.6, 2.8, facecolor="#e2e8f0", edgecolor="#94a3b8", lw=1.8, alpha=0.32)
    ax.add_patch(left_plate)
    ax.add_patch(right_plate)
    ax.text(-4.95, 1.12, "chasis fijo izquierdo", fontsize=11, color="#475569")
    ax.text(2.42, 1.12, "chasis fijo derecho", fontsize=11, color="#475569")

    ax.plot([-5.3, 5.3], [1.7, 1.7], color="#111827", lw=3.0)
    ax.scatter([-3.0, 3.0], [1.7, 1.7], s=70, color="#111827", zorder=4)
    ax.text(-3.18, 1.95, "O2_L", fontsize=11, color="#111827")
    ax.text(2.82, 1.95, "O2_R", fontsize=11, color="#111827")
    ax.text(-0.55, 1.98, "eje motriz común", fontsize=11, color="#111827")

    motor = plt.Circle((0.0, 1.7), 0.42, facecolor="#f59e0b", edgecolor="#7c2d12", lw=2.0)
    ax.add_patch(motor)
    ax.text(0.0, 1.7, "motor", ha="center", va="center", fontsize=11, color="#111827")

    o4_l = (-5.0, 1.0)
    o2_l = (-3.0, 1.0)
    o6_l = (-4.45, -0.95)
    a_l = (-2.15, 0.15)
    b_l = (-4.05, -0.25)
    p_l = (-3.25, -0.58)
    q_l = (-3.95, -1.00)
    left_plate_poly = plt.Polygon([a_l, b_l, p_l], closed=True, facecolor="#c4b5fd", edgecolor="none", alpha=0.28)
    ax.add_patch(left_plate_poly)
    ax.plot([o4_l[0], o2_l[0]], [o4_l[1], o2_l[1]], color="#111827", lw=2.4, ls=":")
    ax.plot([o2_l[0], a_l[0]], [o2_l[1], a_l[1]], color="#2563eb", lw=3.0)
    ax.plot([a_l[0], b_l[0]], [a_l[1], b_l[1]], color="#dc2626", lw=3.0)
    ax.plot([b_l[0], o4_l[0]], [b_l[1], o4_l[1]], color="#059669", lw=3.0)
    ax.plot([a_l[0], p_l[0]], [a_l[1], p_l[1]], color="#7c3aed", lw=2.6, ls="--")
    ax.plot([b_l[0], p_l[0]], [b_l[1], p_l[1]], color="#7c3aed", lw=2.0, ls="--")
    ax.plot([p_l[0], q_l[0]], [p_l[1], q_l[1]], color="#ea580c", lw=3.0)
    ax.plot([o6_l[0], q_l[0]], [o6_l[1], q_l[1]], color="#92400e", lw=3.0)
    ax.scatter(
        [o4_l[0], o2_l[0], o6_l[0], a_l[0], b_l[0], p_l[0], q_l[0]],
        [o4_l[1], o2_l[1], o6_l[1], a_l[1], b_l[1], p_l[1], q_l[1]],
        s=55,
        color="#111827",
    )
    ax.text(o4_l[0] - 0.15, o4_l[1] + 0.18, "O4_L", fontsize=10)
    ax.text(o6_l[0] - 0.18, o6_l[1] - 0.22, "O6_L", fontsize=10)
    ax.text(a_l[0] + 0.08, a_l[1] + 0.12, "A_L", fontsize=10, color="#2563eb")
    ax.text(b_l[0] - 0.02, b_l[1] - 0.2, "B_L", fontsize=10, color="#dc2626")
    ax.text(p_l[0] + 0.05, p_l[1] - 0.2, "P_L", fontsize=10, color="#7c3aed")
    ax.text(q_l[0] - 0.08, q_l[1] - 0.22, "Q_L", fontsize=10, color="#ea580c")
    ax.annotate("", xy=(-2.4, 0.65), xytext=(-2.85, 1.35), arrowprops=dict(arrowstyle="->", color="#2563eb", lw=2))
    ax.text(-2.55, 1.35, r"$\omega$", fontsize=12, color="#2563eb")
    ax.text(-5.15, -1.72, "módulo izquierdo\n6 barras", fontsize=11, color="#111827")
    left_traj_x = np.array([-3.7, -3.35, -3.0, -2.75, -2.95, -3.45, -3.8])
    left_traj_y = np.array([-0.95, -0.55, -0.1, -0.25, -0.75, -1.0, -0.95])
    ax.plot(left_traj_x, left_traj_y, color="#94a3b8", lw=2.0)

    o2_r = (3.0, 1.0)
    o4_r = (5.0, 1.0)
    o6_r = (4.45, -0.95)
    a_r = (2.15, 0.15)
    b_r = (4.05, -0.25)
    p_r = (3.25, -0.58)
    q_r = (3.95, -1.00)
    right_plate_poly = plt.Polygon([a_r, b_r, p_r], closed=True, facecolor="#c4b5fd", edgecolor="none", alpha=0.28)
    ax.add_patch(right_plate_poly)
    ax.plot([o2_r[0], o4_r[0]], [o2_r[1], o4_r[1]], color="#111827", lw=2.4, ls=":")
    ax.plot([o2_r[0], a_r[0]], [o2_r[1], a_r[1]], color="#2563eb", lw=3.0)
    ax.plot([a_r[0], b_r[0]], [a_r[1], b_r[1]], color="#dc2626", lw=3.0)
    ax.plot([b_r[0], o4_r[0]], [b_r[1], o4_r[1]], color="#059669", lw=3.0)
    ax.plot([a_r[0], p_r[0]], [a_r[1], p_r[1]], color="#7c3aed", lw=2.6, ls="--")
    ax.plot([b_r[0], p_r[0]], [b_r[1], p_r[1]], color="#7c3aed", lw=2.0, ls="--")
    ax.plot([p_r[0], q_r[0]], [p_r[1], q_r[1]], color="#ea580c", lw=3.0)
    ax.plot([o6_r[0], q_r[0]], [o6_r[1], q_r[1]], color="#92400e", lw=3.0)
    ax.scatter(
        [o4_r[0], o2_r[0], o6_r[0], a_r[0], b_r[0], p_r[0], q_r[0]],
        [o4_r[1], o2_r[1], o6_r[1], a_r[1], b_r[1], p_r[1], q_r[1]],
        s=55,
        color="#111827",
    )
    ax.text(o4_r[0] - 0.15, o4_r[1] + 0.18, "O4_R", fontsize=10)
    ax.text(o6_r[0] - 0.18, o6_r[1] - 0.22, "O6_R", fontsize=10)
    ax.text(a_r[0] - 0.22, a_r[1] + 0.12, "A_R", fontsize=10, color="#2563eb")
    ax.text(b_r[0] - 0.02, b_r[1] - 0.2, "B_R", fontsize=10, color="#dc2626")
    ax.text(p_r[0] - 0.02, p_r[1] - 0.2, "P_R", fontsize=10, color="#7c3aed")
    ax.text(q_r[0] - 0.08, q_r[1] - 0.22, "Q_R", fontsize=10, color="#ea580c")
    ax.annotate("", xy=(2.4, 0.65), xytext=(2.85, 1.35), arrowprops=dict(arrowstyle="->", color="#2563eb", lw=2))
    ax.text(2.35, 1.35, r"$\omega + \Delta\phi$", fontsize=12, color="#2563eb")
    ax.text(3.15, -1.72, "módulo derecho\n6 barras", fontsize=11, color="#111827")
    right_traj_x = np.array([3.8, 3.45, 3.05, 2.8, 3.0, 3.5, 3.85])
    right_traj_y = np.array([-0.95, -0.55, -0.1, -0.25, -0.75, -1.0, -0.95])
    ax.plot(right_traj_x, right_traj_y, color="#94a3b8", lw=2.0)

    ax.text(0.0, 0.18, "P = punto intermedio del acoplador", ha="center", fontsize=11, color="#7c3aed")
    ax.text(0.0, -0.12, "Q = pasador comun entre los eslabones P-Q y O6-Q", ha="center", fontsize=11, color="#111827")
    ax.text(0.0, -0.42, "Si faltara P-Q, O6-Q si seria un pendulo; con ambos, Q queda restringido", ha="center", fontsize=11, color="#ea580c")
    ax.text(0.0, -0.72, "O2, O4 y O6 son pivotes fijos anclados al chasis", ha="center", fontsize=11, color="#334155")

    ax.text(
        0.0,
        -2.18,
        "Arquitectura recomendada con un solo motor: dos mecanismos 6 barras anclados a chasis,\n"
        "uno por brazo, acoplados a un mismo eje y con desfase de fase entre lados.",
        ha="center",
        fontsize=12,
        color="#111827",
    )
    ax.text(
        0.0,
        -2.62,
        "Cada modulo sigue siendo un mecanismo cerrado de 1 GDL. La salida distal no cuelga libre:\n"
        "queda restringida geometricamente por la diada adicional y por eso no se comporta como pendulo pasivo.",
        ha="center",
        fontsize=11,
        color="#334155",
    )

    fig.tight_layout()
    fig.savefig(FIGURE_SPECS["dual_architecture"], dpi=180, bbox_inches="tight")
    plt.close(fig)


def interpolate_open_curve(
    sample_phase: np.ndarray,
    sample_xy: np.ndarray,
    fine_phase: np.ndarray,
) -> np.ndarray:
    x_spline = CubicSpline(sample_phase, sample_xy[:, 0], bc_type="natural")
    y_spline = CubicSpline(sample_phase, sample_xy[:, 1], bc_type="natural")
    return np.column_stack([x_spline(fine_phase), y_spline(fine_phase)])


def build_mujoco_targets(
    body_track: dict[str, np.ndarray],
    distal_records: dict[str, np.ndarray],
    n_samples: int = 121,
) -> dict[str, np.ndarray]:
    analyzed_stop = int(DISTAL_TRACK_FRAMES[-1]) + 1
    mean_bl = float(np.mean(body_track["body_length_px"][:analyzed_stop]))
    body_phase = np.linspace(0.0, 1.0, analyzed_stop)
    fine_phase = np.linspace(0.0, 1.0, n_samples)

    body_x = MUJOCO_BODY_X_SCALE * (
        body_track["center_x"][:analyzed_stop] - body_track["center_x"][0]
    ) / mean_bl
    body_z = MUJOCO_BODY_Z_OFFSET + MUJOCO_BODY_Z_SCALE * (
        -(body_track["center_y"][:analyzed_stop] - body_track["center_y"][0]) / mean_bl
    )
    body_xz = interpolate_open_curve(body_phase, np.column_stack([body_x, body_z]), fine_phase)
    body_vel = np.gradient(body_xz, fine_phase, axis=0)
    body_pitch = np.clip(
        -0.14 * np.arctan2(body_vel[:, 1], body_vel[:, 0] + 1e-9),
        -0.18,
        0.18,
    )

    tip_phase = distal_records["frame"] / float(DISTAL_TRACK_FRAMES[-1])
    target_a_local = interpolate_open_curve(
        tip_phase,
        np.column_stack([distal_records["primary_tip_x_bl"], distal_records["primary_tip_y_bl"]]),
        fine_phase,
    )
    target_b_local = interpolate_open_curve(
        tip_phase,
        np.column_stack([distal_records["secondary_tip_x_bl"], distal_records["secondary_tip_y_bl"]]),
        fine_phase,
    )

    return {
        "phase": fine_phase,
        "body_xz": body_xz,
        "body_pitch": body_pitch,
        "target_a_local": target_a_local,
        "target_b_local": target_b_local,
        "target_a_world": body_xz + target_a_local,
        "target_b_world": body_xz + target_b_local,
    }


def initialize_tentacle_joints(
    base_xy: np.ndarray,
    target_xy: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    target_xy = np.asarray(target_xy, dtype=float)
    base_xy = np.asarray(base_xy, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    total_length = float(np.sum(lengths))
    direction = target_xy - base_xy
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-9:
        direction = np.array([1.0, 0.0], dtype=float)
        direction_norm = 1.0

    joints = [base_xy.copy()]
    prev = base_xy.copy()
    for idx, length in enumerate(lengths, start=1):
        s = float(np.sum(lengths[:idx]) / total_length)
        desired = (1.0 - s) * base_xy + s * target_xy + np.array(
            [0.0, -0.22 * total_length * 4.0 * s * (1.0 - s)],
            dtype=float,
        )
        vec = desired - prev
        vec_norm = float(np.linalg.norm(vec))
        if vec_norm < 1e-9:
            vec = direction
            vec_norm = direction_norm
        prev = prev + length * vec / vec_norm
        joints.append(prev.copy())
    return np.asarray(joints, dtype=float)


def solve_tentacle_chain(
    target_xy: np.ndarray,
    base_xy: np.ndarray,
    lengths: np.ndarray,
    joints_seed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    target_xy = np.asarray(target_xy, dtype=float)
    base_xy = np.asarray(base_xy, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    total_length = float(np.sum(lengths))
    direction = target_xy - base_xy
    distance = float(np.linalg.norm(direction))

    if distance > total_length * 0.995 and distance > 1e-9:
        target_xy = base_xy + direction * ((total_length * 0.995) / distance)
        direction = target_xy - base_xy
        distance = float(np.linalg.norm(direction))

    if joints_seed is None:
        joints = initialize_tentacle_joints(base_xy, target_xy, lengths)
    else:
        joints = np.asarray(joints_seed, dtype=float).copy()
        joints += base_xy - joints[0]

    if distance >= total_length * 0.995:
        direction_unit = direction / max(distance, 1e-9)
        joints[0] = base_xy
        for idx, length in enumerate(lengths, start=1):
            joints[idx] = joints[idx - 1] + length * direction_unit
    else:
        base_anchor = base_xy.copy()
        for _ in range(80):
            joints[-1] = target_xy
            for idx in range(len(lengths) - 1, -1, -1):
                delta = joints[idx] - joints[idx + 1]
                delta_norm = float(np.linalg.norm(delta))
                if delta_norm < 1e-9:
                    delta = np.array([0.0, -1.0], dtype=float)
                    delta_norm = 1.0
                joints[idx] = joints[idx + 1] + lengths[idx] * delta / delta_norm

            joints[0] = base_anchor
            for idx in range(len(lengths)):
                delta = joints[idx + 1] - joints[idx]
                delta_norm = float(np.linalg.norm(delta))
                if delta_norm < 1e-9:
                    delta = np.array([1.0, 0.0], dtype=float)
                    delta_norm = 1.0
                joints[idx + 1] = joints[idx] + lengths[idx] * delta / delta_norm

            if np.linalg.norm(joints[-1] - target_xy) < 1e-4:
                break

    vectors = np.diff(joints, axis=0)
    absolute_angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    q = np.empty(len(lengths), dtype=float)
    q[0] = absolute_angles[0]
    q[1:] = np.diff(absolute_angles)
    return q, joints


def build_tentacle_xml(prefix: str, base_xy: np.ndarray, lateral_offset: float) -> str:
    colors = [
        "0.83 0.60 0.38 1",
        "0.80 0.55 0.33 1",
        "0.77 0.51 0.31 1",
        "0.74 0.48 0.30 1",
        "0.72 0.46 0.29 1",
        "0.70 0.45 0.28 1",
        "0.68 0.44 0.27 1",
    ]
    radii = [0.028, 0.026, 0.024, 0.022, 0.020, 0.018, 0.016]
    parts = [f'<body name="{prefix}_base" pos="{base_xy[0]} {lateral_offset} {base_xy[1]}">']
    for idx, (length, radius, color) in enumerate(zip(MUJOCO_ARM_LINKS_BL, radii, colors), start=1):
        parts.append(f'<joint name="{prefix}_q{idx}" type="hinge" axis="0 1 0"/>')
        parts.append(
            f'<geom type="capsule" fromto="0 0 0 {length} 0 0" size="{radius}" rgba="{color}"/>'
        )
        if idx < len(MUJOCO_ARM_LINKS_BL):
            parts.append(f'<body name="{prefix}_link{idx + 1}" pos="{length} 0 0">')
    parts.append(f'<site name="{prefix}_tip" pos="{MUJOCO_ARM_LINKS_BL[-1]} 0 0" size="0.012"/>')
    for _ in MUJOCO_ARM_LINKS_BL:
        parts.append("</body>")
    return "\n".join(parts)


def build_mocap_capsule_chain_xml(
    prefix: str,
    lengths: np.ndarray,
    radii: np.ndarray,
    colors: list[str],
) -> str:
    parts = []
    for idx, (length, radius, color) in enumerate(zip(lengths, radii, colors), start=1):
        parts.append(f'<body name="{prefix}_{idx}" mocap="true">')
        parts.append(
            f'<geom type="capsule" fromto="0 0 0 {float(length)} 0 0" size="{float(radius)}" rgba="{color}"/>'
        )
        parts.append("</body>")
    return "\n".join(parts)


def build_mocap_sphere_markers_xml(prefix: str, count: int, size: float, color: str) -> str:
    parts = []
    for idx in range(count):
        parts.append(f'<body name="{prefix}_{idx}" mocap="true">')
        parts.append(f'<geom type="sphere" size="{float(size)}" rgba="{color}"/>')
        parts.append("</body>")
    return "\n".join(parts)


def set_capsule_chain_poses(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    prefix: str,
    joints_xy: np.ndarray,
    lateral_offset: float,
) -> None:
    for idx, (p0, p1) in enumerate(zip(joints_xy[:-1], joints_xy[1:]), start=1):
        delta = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
        angle = math.atan2(float(delta[1]), float(delta[0]))
        set_mocap_pose(data, model, f"{prefix}_{idx}", np.asarray(p0, dtype=float), angle, lateral_offset=lateral_offset)


def set_marker_positions(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    prefix: str,
    points_xy: np.ndarray,
    lateral_offset: float,
) -> None:
    for idx, point in enumerate(points_xy):
        point_array = np.asarray(point, dtype=float)
        if point_array.shape[0] >= 3:
            mocap_id = body_mocap_id(model, f"{prefix}_{idx}")
            data.mocap_pos[mocap_id] = np.array(
                [float(point_array[0]), float(point_array[1]), float(point_array[2])],
                dtype=float,
            )
            data.mocap_quat[mocap_id] = y_axis_quat(0.0)
        else:
            set_mocap_pose(data, model, f"{prefix}_{idx}", point_array, 0.0, lateral_offset=lateral_offset)


def cubic_bezier_points(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    n_points: int,
) -> np.ndarray:
    points = []
    for t in np.linspace(0.0, 1.0, n_points):
        omt = 1.0 - t
        point = (
            (omt**3) * p0
            + 3.0 * (omt**2) * t * p1
            + 3.0 * omt * (t**2) * p2
            + (t**3) * p3
        )
        points.append(point)
    return np.asarray(points, dtype=float)


def rotate_xy(points: np.ndarray, angle_rad: float, origin: np.ndarray) -> np.ndarray:
    points_array = np.asarray(points, dtype=float)
    origin_array = np.asarray(origin, dtype=float)
    rot = np.array(
        [
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)],
        ],
        dtype=float,
    )
    if points_array.ndim == 1:
        return rot @ (points_array - origin_array)
    return (points_array - origin_array) @ rot.T


def annotate_mujoco_frame(image: Image.Image) -> Image.Image:
    frame = image.convert("RGBA")
    draw = ImageDraw.Draw(frame, "RGBA")
    panel = (18, 18, 265, 56)
    draw.rounded_rectangle(panel, radius=12, fill=(255, 255, 255, 218), outline=(203, 213, 225, 255), width=2)
    draw.text((32, 31), "MuJoCo: 8 barras (MotionGen)", fill=(15, 23, 42, 255), anchor="lm")
    return frame.convert("RGB")


def create_mujoco_match_plot(mujoco_solution: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.plot(
        mujoco_solution["raw_target_xy"][:, 0],
        mujoco_solution["raw_target_xy"][:, 1],
        color="#0f766e",
        lw=2.4,
        label="Trayectoria real del video",
    )
    ax.plot(
        mujoco_solution["design_target_xy"][:, 0],
        mujoco_solution["design_target_xy"][:, 1],
        color="#1d4ed8",
        lw=2.0,
        ls="--",
        label="Objetivo simplificado",
    )
    trace_xy = np.asarray(mujoco_solution["trace_xy"], dtype=float)
    if trace_xy.size:
        ax.plot(
            trace_xy[:, 0],
            trace_xy[:, 1],
            color="#dc2626",
            lw=1.8,
            ls="-.",
            label="Trayectoria del punto G en MuJoCo",
        )
    ax.set_title("Trayectoria real, aproximada y seguida por el 8 barras en MuJoCo")
    ax.set_xlabel("x [BL]")
    ax.set_ylabel("y [BL]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_SPECS["mujoco_match"], dpi=180, bbox_inches="tight")
    plt.close(fig)


def motiongen_pair_targets() -> list[tuple[str, str, str, float]]:
    pair_targets: list[tuple[str, str, str, float]] = []
    for body_name, node_names in MOTIONGEN_8BAR_BODIES.items():
        nodes = list(node_names)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a_name = nodes[i]
                b_name = nodes[j]
                pair_targets.append(
                    (
                        body_name,
                        a_name,
                        b_name,
                        float(
                            np.linalg.norm(
                                MOTIONGEN_8BAR_POINTS_BL[a_name] - MOTIONGEN_8BAR_POINTS_BL[b_name]
                            )
                        ),
                    )
                )
    return pair_targets


def rotate_closed_cycle(xy: np.ndarray, start_idx: int) -> np.ndarray:
    core = np.asarray(xy[:-1], dtype=float)
    rotated = np.vstack([core[start_idx:], core[:start_idx]])
    return np.vstack([rotated, rotated[0]])


def motiongen_unpack_state(
    vector: np.ndarray,
    target_point_xy: np.ndarray,
    movable_nodes: list[str],
) -> dict[str, np.ndarray]:
    state = {
        node_name: MOTIONGEN_8BAR_POINTS_BL[node_name].copy()
        for node_name in MOTIONGEN_FIXED_NODES
    }
    state[MOTIONGEN_OUTPUT_NODE] = np.asarray(target_point_xy, dtype=float).copy()
    for node_idx, node_name in enumerate(movable_nodes):
        state[node_name] = np.asarray(vector[2 * node_idx : 2 * node_idx + 2], dtype=float)
    return state


def solve_motiongen_kinematic_path(
    design_target_xy: np.ndarray,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, float]], np.ndarray]:
    design_target_xy = np.asarray(design_target_xy, dtype=float)
    initial_output = MOTIONGEN_8BAR_POINTS_BL[MOTIONGEN_OUTPUT_NODE]
    start_idx = int(np.argmin(np.linalg.norm(design_target_xy[:-1] - initial_output, axis=1)))
    rotated_target_xy = rotate_closed_cycle(design_target_xy, start_idx)
    pair_targets = motiongen_pair_targets()
    movable_nodes = [
        node_name
        for node_name in MOTIONGEN_8BAR_POINTS_BL
        if node_name not in MOTIONGEN_FIXED_NODES and node_name != MOTIONGEN_OUTPUT_NODE
    ]
    previous_vector = np.concatenate(
        [MOTIONGEN_8BAR_POINTS_BL[node_name] for node_name in movable_nodes]
    )
    previous_target = rotated_target_xy[0]
    states: list[dict[str, np.ndarray]] = []
    state_rows: list[dict[str, float]] = []

    for phase_idx, target_xy in enumerate(rotated_target_xy[:-1]):
        previous_state = motiongen_unpack_state(previous_vector, previous_target, movable_nodes)
        initial_guess = previous_vector.copy()
        if phase_idx > 0:
            delta = target_xy - previous_target
            for node_idx in range(len(movable_nodes)):
                initial_guess[2 * node_idx : 2 * node_idx + 2] += 0.35 * delta

        def residual(vector: np.ndarray) -> np.ndarray:
            state = motiongen_unpack_state(vector, target_xy, movable_nodes)
            residual_values = []
            for _, start_name, end_name, target_length in pair_targets:
                residual_values.append(
                    float(np.linalg.norm(state[start_name] - state[end_name]) - target_length)
                )
            for node_name in movable_nodes:
                residual_values.extend(
                    (
                        0.10 * (state[node_name] - previous_state[node_name])
                    ).tolist()
                )
            for node_name, weight in (("A", 0.02), ("B", 0.02), ("I", 0.02), ("K", 0.02)):
                if node_name in movable_nodes:
                    residual_values.extend(
                        (
                            weight
                            * (state[node_name] - MOTIONGEN_8BAR_POINTS_BL[node_name])
                        ).tolist()
                    )
            return np.asarray(residual_values, dtype=float)

        result = least_squares(
            residual,
            initial_guess,
            max_nfev=5000,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        solved_state = motiongen_unpack_state(result.x, target_xy, movable_nodes)
        previous_vector = result.x.copy()
        previous_target = target_xy.copy()
        states.append(solved_state)

        link_errors = [
            float(np.linalg.norm(solved_state[a_name] - solved_state[b_name]) - target_length)
            for _, a_name, b_name, target_length in pair_targets
        ]
        row = {
            "phase_fraction": float(phase_idx / max(len(rotated_target_xy) - 2, 1)),
            "output_x_bl": float(solved_state[MOTIONGEN_OUTPUT_NODE][0]),
            "output_y_bl": float(solved_state[MOTIONGEN_OUTPUT_NODE][1]),
            "link_rms_error_bl": float(np.sqrt(np.mean(np.square(link_errors)))),
        }
        for node_name, node_xy in solved_state.items():
            row[f"{node_name}_x_bl"] = float(node_xy[0])
            row[f"{node_name}_y_bl"] = float(node_xy[1])
        state_rows.append(row)

    return states, state_rows, rotated_target_xy


def world_from_bl(point_xy: np.ndarray) -> np.ndarray:
    point_xy = np.asarray(point_xy, dtype=float)
    return np.array([float(point_xy[0]), 0.0, float(point_xy[1])], dtype=float)


def capsule_pose_from_points(point_start_xy: np.ndarray, point_end_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    p0 = world_from_bl(point_start_xy)
    p1 = world_from_bl(point_end_xy)
    center = 0.5 * (p0 + p1)
    direction = p1 - p0
    length = float(np.linalg.norm(direction))
    angle_y = float(np.arctan2(direction[0], direction[2] + 1e-12))
    quat = y_axis_quat(angle_y)
    return center, quat, length


def qpos_address(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_qposadr[joint_id])


def set_free_body_pose(data: mujoco.MjData, qpos_addr: int, center_world: np.ndarray, quat: np.ndarray) -> None:
    data.qpos[qpos_addr : qpos_addr + 3] = center_world
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat


def build_mujoco_gif(
    reference_cycle: dict[str, np.ndarray],
    design_cycle: dict[str, np.ndarray | float | int],
    fourbar_solution: dict[str, np.ndarray | float | int],
) -> dict[str, np.ndarray]:
    del fourbar_solution
    raw_target_xy = np.asarray(reference_cycle["xy"], dtype=float)
    design_target_xy = np.asarray(design_cycle["xy"], dtype=float)

    states, state_rows, rotated_design_xy = solve_motiongen_kinematic_path(design_target_xy)
    initial_output = MOTIONGEN_8BAR_POINTS_BL[MOTIONGEN_OUTPUT_NODE]
    start_idx = int(np.argmin(np.linalg.norm(design_target_xy[:-1] - initial_output, axis=1)))
    rotated_raw_xy = rotate_closed_cycle(raw_target_xy, start_idx)
    trace_xy = np.vstack(
        [
            np.asarray([state[MOTIONGEN_OUTPUT_NODE] for state in states], dtype=float),
            np.asarray(states[0][MOTIONGEN_OUTPUT_NODE], dtype=float)[None, :],
        ]
    )
    phase = np.linspace(0.0, 1.0, len(rotated_design_xy))

    motiongen_rows = []
    for link_idx, (start_name, end_name) in enumerate(MOTIONGEN_8BAR_VIEW_EDGES, start=1):
        start_xy = MOTIONGEN_8BAR_POINTS_BL[start_name]
        end_xy = MOTIONGEN_8BAR_POINTS_BL[end_name]
        motiongen_rows.append(
            {
                "link_id": f"L{link_idx}",
                "joint_i": start_name,
                "joint_j": end_name,
                "x_i_bl": float(start_xy[0]),
                "y_i_bl": float(start_xy[1]),
                "x_j_bl": float(end_xy[0]),
                "y_j_bl": float(end_xy[1]),
                "length_bl": float(np.linalg.norm(end_xy - start_xy)),
            }
        )

    stacked_states = np.vstack(
        [
            np.asarray([state[node_name] for state in states for node_name in state], dtype=float),
            rotated_raw_xy[:-1],
            rotated_design_xy[:-1],
        ]
    )
    min_x, max_x = float(np.min(stacked_states[:, 0])), float(np.max(stacked_states[:, 0]))
    min_z, max_z = float(np.min(stacked_states[:, 1])), float(np.max(stacked_states[:, 1]))
    pad_x = 0.12 * max(max_x - min_x, 1e-6)
    pad_z = 0.12 * max(max_z - min_z, 1e-6)
    min_x -= pad_x
    max_x += pad_x
    min_z -= pad_z
    max_z += pad_z
    center_x = 0.5 * (min_x + max_x)
    center_z = 0.5 * (min_z + max_z)
    extent = max(max_x - min_x, max_z - min_z)
    workspace_center = np.array([center_x, center_z], dtype=float)
    workspace_size = 0.5 * np.array([max_x - min_x, max_z - min_z], dtype=float)

    edge_specs = [
        {
            "name": f"{start_name}_{end_name}",
            "start": start_name,
            "end": end_name,
            "radius": 0.013 if (start_name, end_name) in {("F", "G"), ("D", "B")} else 0.011,
            "rgba": (0.44, 0.93, 0.64, 0.98),
        }
        for start_name, end_name in MOTIONGEN_8BAR_VIEW_EDGES
    ]
    edge_lengths = {
        spec["name"]: float(
            np.linalg.norm(
                MOTIONGEN_8BAR_POINTS_BL[spec["end"]] - MOTIONGEN_8BAR_POINTS_BL[spec["start"]]
            )
        )
        for spec in edge_specs
    }

    raw_markers = []
    for point in rotated_raw_xy[::6]:
        raw_markers.append(
            f'<geom type="sphere" pos="{point[0]} 0 {point[1]}" size="0.0062" rgba="0.07 0.45 0.43 0.90"/>'
        )
    design_markers = []
    for point in rotated_design_xy[::6]:
        design_markers.append(
            f'<geom type="sphere" pos="{point[0]} 0 {point[1]}" size="0.0056" rgba="0.13 0.40 0.93 0.95"/>'
        )

    body_blocks = []
    for spec in edge_specs:
        half_length = 0.5 * edge_lengths[spec["name"]]
        rgba = " ".join(f"{value:.6f}" for value in spec["rgba"])
        body_blocks.append(
            f"""
        <body name="edge_{spec['name']}" pos="0 0 0">
          <freejoint name="joint_edge_{spec['name']}"/>
          <geom name="geom_edge_{spec['name']}" type="capsule" size="{spec['radius']:.6f} {half_length:.6f}" rgba="{rgba}"/>
        </body>
"""
        )

    for joint_name in MOTIONGEN_8BAR_POINTS_BL:
        rgba = "0.98 0.87 0.15 1" if joint_name != MOTIONGEN_OUTPUT_NODE else "0.71 0.52 0.96 1"
        radius = 0.018 if joint_name != MOTIONGEN_OUTPUT_NODE else 0.024
        body_blocks.append(
            f"""
        <body name="mark_{joint_name}" pos="0 0 0">
          <freejoint name="joint_mark_{joint_name}"/>
          <geom name="geom_mark_{joint_name}" type="sphere" size="{radius:.6f}" rgba="{rgba}"/>
        </body>
"""
        )

    camera_y = max(1.05, 1.55 * extent)
    xml = f"""
    <mujoco model="motiongen_8bar_kinematic">
      <compiler angle="radian"/>
      <option gravity="0 0 0" timestep="0.01"/>
      <default>
        <geom contype="0" conaffinity="0"/>
      </default>
      <visual>
        <global offwidth="1200" offheight="900"/>
        <headlight diffuse="0.90 0.90 0.90" ambient="0.28 0.28 0.28"/>
        <rgba haze="0.98 0.98 0.98 1"/>
      </visual>
      <worldbody>
        <light pos="{center_x} 0.6 {center_z + 0.4}" dir="0 -0.3 -1" directional="true"/>
        <camera name="cam_main" pos="{center_x:.6f} {-camera_y:.6f} {center_z:.6f}" xyaxes="1 0 0 0 0 1"/>
        <geom type="box" pos="{workspace_center[0]:.6f} 0 {workspace_center[1]:.6f}" size="{workspace_size[0]:.6f} 0.0018 {workspace_size[1]:.6f}" rgba="0.88 0.89 0.91 0.18"/>
        {''.join(raw_markers)}
        {''.join(design_markers)}
        {''.join(body_blocks)}
      </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=1200, height=900)
    edge_qpos = {
        spec["name"]: qpos_address(model, f"joint_edge_{spec['name']}")
        for spec in edge_specs
    }
    joint_qpos = {
        joint_name: qpos_address(model, f"joint_mark_{joint_name}")
        for joint_name in MOTIONGEN_8BAR_POINTS_BL
    }

    frames: list[Image.Image] = []
    for state in states:
        for spec in edge_specs:
            point_start = state[spec["start"]]
            point_end = state[spec["end"]]
            center_world, quat, _ = capsule_pose_from_points(point_start, point_end)
            set_free_body_pose(data, edge_qpos[spec["name"]], center_world, quat)
        for joint_name, point_xy in state.items():
            set_free_body_pose(
                data,
                joint_qpos[joint_name],
                world_from_bl(point_xy),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="cam_main")
        frames.append(annotate_mujoco_frame(Image.fromarray(renderer.render())))

    frames[0].save(FIGURE_SPECS["motiongen_mujoco_frame"])
    frames[0].save(FIGURE_SPECS["mujoco_frame"])
    frames[0].save(
        MUJOCO_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=55,
        loop=0,
    )
    renderer.close()

    try:
        if MOTIONGEN_EIGHTBAR_CSV.exists():
            MOTIONGEN_EIGHTBAR_CSV.unlink()
        save_csv(
            MOTIONGEN_EIGHTBAR_CSV,
            motiongen_rows,
            ["link_id", "joint_i", "joint_j", "x_i_bl", "y_i_bl", "x_j_bl", "y_j_bl", "length_bl"],
        )
        if MOTIONGEN_EIGHTBAR_STATES_CSV.exists():
            MOTIONGEN_EIGHTBAR_STATES_CSV.unlink()
        state_fieldnames = list(state_rows[0].keys()) if state_rows else ["phase_fraction"]
        save_csv(MOTIONGEN_EIGHTBAR_STATES_CSV, state_rows, state_fieldnames)
    except Exception:
        pass

    solution = {
        "phase": phase,
        "raw_target_xy": rotated_raw_xy,
        "design_target_xy": rotated_design_xy,
        "trace_xy": trace_xy,
        "a_trace_xy": np.empty((0, 2), dtype=float),
        "b_trace_xy": np.empty((0, 2), dtype=float),
        "motiongen_rows": motiongen_rows,
    }
    create_mujoco_match_plot(solution)
    return solution


def y_axis_quat(angle_rad: float) -> np.ndarray:
    return np.array(
        [math.cos(angle_rad / 2.0), 0.0, math.sin(angle_rad / 2.0), 0.0],
        dtype=float,
    )


def body_mocap_id(model: mujoco.MjModel, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise KeyError(f"El body {body_name} no es mocap.")
    return mocap_id


def set_mocap_pose(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    body_name: str,
    xy: np.ndarray,
    angle_rad: float = 0.0,
    lateral_offset: float = 0.0,
) -> None:
    mocap_id = body_mocap_id(model, body_name)
    data.mocap_pos[mocap_id] = np.array([float(xy[0]), float(lateral_offset), float(xy[1])], dtype=float)
    data.mocap_quat[mocap_id] = y_axis_quat(float(angle_rad))


def build_sixbar_mujoco_gif(
    reference_cycle: dict[str, np.ndarray],
    sixbar_solution: dict[str, np.ndarray | float | int],
) -> None:
    params = np.asarray(sixbar_solution["params"], dtype=float)
    x0, y0, phi, d, a, b, c, px, py, theta_offset, o6x, o6y, pq_len, o6q_len = params
    o2 = np.array([x0, y0], dtype=float)
    o4 = o2 + d * np.array([math.cos(phi), math.sin(phi)], dtype=float)
    o6 = np.array([o6x, o6y], dtype=float)
    phase = reference_cycle["phase"]
    target_xy = reference_cycle["xy"]
    q_trace = np.asarray(sixbar_solution["q_trace"], dtype=float)
    stacked = np.vstack([target_xy, q_trace, o2[None, :], o4[None, :], o6[None, :]])
    min_x, max_x = float(np.min(stacked[:, 0])), float(np.max(stacked[:, 0]))
    min_z, max_z = float(np.min(stacked[:, 1])), float(np.max(stacked[:, 1]))
    center_x = 0.5 * (min_x + max_x)
    center_z = 0.5 * (min_z + max_z)
    extent = max(max_x - min_x, max_z - min_z)
    plane_z = min_z - 0.28

    target_markers = []
    for point in target_xy[::6]:
        target_markers.append(
            f'<geom type="sphere" pos="{float(point[0])} 0 {float(point[1])}" size="0.010" rgba="0.06 0.46 0.43 0.55"/>'
        )
    xml = f"""
    <mujoco model="sixbar_mechanism">
      <compiler angle="radian"/>
      <option timestep="0.01" gravity="0 0 0"/>
      <visual>
        <global offwidth="960" offheight="700"/>
        <headlight diffuse="0.9 0.9 0.9" ambient="0.35 0.35 0.35"/>
      </visual>
      <worldbody>
        <light pos="0 0 2.2" dir="0 0 -1"/>
        <geom type="plane" size="4 4 0.1" pos="0 0 {plane_z}" rgba="0.95 0.95 0.94 1"/>
        <geom type="sphere" pos="{o2[0]} 0 {o2[1]}" size="0.020" rgba="0.07 0.09 0.16 1"/>
        <geom type="sphere" pos="{o4[0]} 0 {o4[1]}" size="0.020" rgba="0.07 0.09 0.16 1"/>
        <geom type="sphere" pos="{o6[0]} 0 {o6[1]}" size="0.020" rgba="0.07 0.09 0.16 1"/>
        <geom type="capsule" fromto="{o2[0]} 0 {o2[1]} {o4[0]} 0 {o4[1]}" size="0.010" rgba="0.12 0.12 0.15 0.95"/>
        {''.join(target_markers)}
        <body name="input_link" mocap="true">
          <geom type="capsule" fromto="0 0 0 {a} 0 0" size="0.016" rgba="0.15 0.39 0.92 1"/>
        </body>
        <body name="output_link" mocap="true">
          <geom type="capsule" fromto="0 0 0 {c} 0 0" size="0.016" rgba="0.02 0.59 0.41 1"/>
        </body>
        <body name="coupler_plate" mocap="true">
          <geom type="capsule" fromto="0 0 0 {b} 0 0" size="0.015" rgba="0.86 0.15 0.15 1"/>
          <geom type="capsule" fromto="0 0 0 {px} 0 {py}" size="0.010" rgba="0.49 0.23 0.93 0.95"/>
          <geom type="capsule" fromto="{b} 0 0 {px} 0 {py}" size="0.010" rgba="0.49 0.23 0.93 0.95"/>
        </body>
        <body name="pq_link" mocap="true">
          <geom type="capsule" fromto="0 0 0 {pq_len} 0 0" size="0.014" rgba="0.92 0.34 0.03 1"/>
        </body>
        <body name="o6q_link" mocap="true">
          <geom type="capsule" fromto="0 0 0 {o6q_len} 0 0" size="0.014" rgba="0.57 0.25 0.05 1"/>
        </body>
        <body name="joint_A" mocap="true">
          <geom type="sphere" size="0.015" rgba="0.07 0.09 0.16 1"/>
        </body>
        <body name="joint_B" mocap="true">
          <geom type="sphere" size="0.015" rgba="0.07 0.09 0.16 1"/>
        </body>
        <body name="joint_P" mocap="true">
          <geom type="sphere" size="0.015" rgba="0.49 0.23 0.93 1"/>
        </body>
        <body name="joint_Q" mocap="true">
          <geom type="sphere" size="0.016" rgba="0.92 0.34 0.03 1"/>
        </body>
      </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=860, height=620)
    camera = mujoco.MjvCamera()
    camera.azimuth = 90.0
    camera.elevation = 6.0
    camera.distance = max(1.8, 1.9 * extent)
    camera.lookat = np.array([center_x, 0.0, center_z], dtype=float)

    frames_rgb: list[Image.Image] = []
    fourbar_branch = int(sixbar_solution["fourbar_branch"])
    dyad_branch = int(sixbar_solution["dyad_branch"])
    for phase_value in phase:
        state = sixbar_state(params, float(phase_value), fourbar_branch=fourbar_branch, dyad_branch=dyad_branch)
        if state is None:
            continue
        o2_point = np.asarray(state["o2"], dtype=float)
        a_point = np.asarray(state["a"], dtype=float)
        b_point = np.asarray(state["b"], dtype=float)
        o4_point = np.asarray(state["o4"], dtype=float)
        p_point = np.asarray(state["p"], dtype=float)
        o6_point = np.asarray(state["o6"], dtype=float)
        q_point = np.asarray(state["q"], dtype=float)

        set_mocap_pose(data, model, "input_link", o2_point, float(state["theta2"]))
        set_mocap_pose(data, model, "output_link", o4_point, float(math.atan2(b_point[1] - o4_point[1], b_point[0] - o4_point[0])))
        set_mocap_pose(data, model, "coupler_plate", a_point, float(state["theta3"]))
        set_mocap_pose(data, model, "pq_link", p_point, float(state["theta_pq"]))
        set_mocap_pose(data, model, "o6q_link", o6_point, float(state["theta_o6q"]))
        set_mocap_pose(data, model, "joint_A", a_point)
        set_mocap_pose(data, model, "joint_B", b_point)
        set_mocap_pose(data, model, "joint_P", p_point)
        set_mocap_pose(data, model, "joint_Q", q_point)

        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frames_rgb.append(Image.fromarray(renderer.render()))

    frames_rgb[0].save(
        SIXBAR_MUJOCO_GIF,
        save_all=True,
        append_images=frames_rgb[1:],
        duration=55,
        loop=0,
    )
    frames_rgb[0].save(FIGURE_SPECS["sixbar_mujoco_frame"])
    renderer.close()


def write_outputs(
    metadata: dict[str, float],
    body_track: dict[str, np.ndarray],
    distal_records: dict[str, np.ndarray],
    reference_cycle: dict[str, np.ndarray],
    design_cycle: dict[str, np.ndarray | float | int],
    angle_solution: dict[str, np.ndarray],
    key_rows: list[dict[str, float | str]],
    fourbar_solution: dict[str, np.ndarray | float | int],
    fourbar_collision_rows: list[dict[str, object]],
    fourbar_collision_summary: dict[str, object],
    fourbar_state_rows: list[dict[str, object]],
    mujoco_solution: dict[str, np.ndarray],
) -> None:
    analyzed_mean_body_length_px = float(
        np.mean(body_track["body_length_px"][: DISTAL_TRACK_FRAMES[-1] + 1])
    )

    body_rows = []
    for row in body_track["rows"]:
        body_rows.append(
            {
                "frame": int(row[0]),
                "time_s": float(row[1]),
                "bbox_x_px": float(row[2]),
                "bbox_y_px": float(row[3]),
                "bbox_w_px": float(row[4]),
                "bbox_h_px": float(row[5]),
                "center_x_px": float(row[6]),
                "center_y_px": float(row[7]),
                "body_length_px": float(row[8]),
            }
        )
    save_csv(
        BODY_TRACK_CSV,
        body_rows,
        ["frame", "time_s", "bbox_x_px", "bbox_y_px", "bbox_w_px", "bbox_h_px", "center_x_px", "center_y_px", "body_length_px"],
    )

    distal_rows = []
    for row in distal_records["rows"]:
        distal_rows.append(
            {
                "frame": int(row[0]),
                "time_s": float(row[1]),
                "phase_fraction": float(row[2]),
                "body_center_x_px": float(row[3]),
                "body_center_y_px": float(row[4]),
                "body_length_px": float(row[5]),
                "primary_tip_x_px": float(row[6]),
                "primary_tip_y_px": float(row[7]),
                "primary_tip_x_bl": float(row[8]),
                "primary_tip_y_bl": float(row[9]),
                "secondary_tip_x_px": float(row[10]),
                "secondary_tip_y_px": float(row[11]),
                "secondary_tip_x_bl": float(row[12]),
                "secondary_tip_y_bl": float(row[13]),
            }
        )
    save_csv(
        DISTAL_TRACK_CSV,
        distal_rows,
        [
            "frame",
            "time_s",
            "phase_fraction",
            "body_center_x_px",
            "body_center_y_px",
            "body_length_px",
            "primary_tip_x_px",
            "primary_tip_y_px",
            "primary_tip_x_bl",
            "primary_tip_y_bl",
            "secondary_tip_x_px",
            "secondary_tip_y_px",
            "secondary_tip_x_bl",
            "secondary_tip_y_bl",
        ],
    )

    cycle_rows = []
    for phase_value, xy, dxy, ddxy in zip(
        reference_cycle["phase"],
        reference_cycle["xy"],
        reference_cycle["dxy"],
        reference_cycle["ddxy"],
    ):
        cycle_rows.append(
            {
                "phase_fraction": float(phase_value),
                "x_bl": float(xy[0]),
                "y_bl": float(xy[1]),
                "vx_bl_per_cycle": float(dxy[0]),
                "vy_bl_per_cycle": float(dxy[1]),
                "ax_bl_per_cycle2": float(ddxy[0]),
                "ay_bl_per_cycle2": float(ddxy[1]),
            }
        )
    save_csv(
        DISTAL_CYCLE_CSV,
        cycle_rows,
        ["phase_fraction", "x_bl", "y_bl", "vx_bl_per_cycle", "vy_bl_per_cycle", "ax_bl_per_cycle2", "ay_bl_per_cycle2"],
    )

    design_rows = []
    for phase_value, xy, dxy, ddxy in zip(
        np.asarray(design_cycle["phase"], dtype=float),
        np.asarray(design_cycle["xy"], dtype=float),
        np.asarray(design_cycle["dxy"], dtype=float),
        np.asarray(design_cycle["ddxy"], dtype=float),
    ):
        design_rows.append(
            {
                "phase_fraction": float(phase_value),
                "x_bl": float(xy[0]),
                "y_bl": float(xy[1]),
                "vx_bl_per_cycle": float(dxy[0]),
                "vy_bl_per_cycle": float(dxy[1]),
                "ax_bl_per_cycle2": float(ddxy[0]),
                "ay_bl_per_cycle2": float(ddxy[1]),
            }
        )
    save_csv(
        DESIGN_CYCLE_CSV,
        design_rows,
        ["phase_fraction", "x_bl", "y_bl", "vx_bl_per_cycle", "vy_bl_per_cycle", "ax_bl_per_cycle2", "ay_bl_per_cycle2"],
    )

    phase_rows = [
        {
            "extremidad": "A",
            "stance_start_pct": RIGHT_STANCE_START * 100.0,
            "stance_end_pct": RIGHT_STANCE_END * 100.0,
            "duty_factor": RIGHT_STANCE_END - RIGHT_STANCE_START,
            "phase_offset_pct": 0.0,
        },
        {
            "extremidad": "B",
            "stance_start_pct": LEFT_STANCE_START * 100.0,
            "stance_end_pct": 100.0,
            "duty_factor": (1.0 - LEFT_STANCE_START) + LEFT_STANCE_WRAP_END,
            "phase_offset_pct": 50.0,
        },
    ]
    save_csv(
        PHASE_CSV,
        phase_rows,
        ["extremidad", "stance_start_pct", "stance_end_pct", "duty_factor", "phase_offset_pct"],
    )

    angle_rows = []
    for phase_value, angles_deg in zip(angle_solution["phase"], angle_solution["angles_deg"]):
        angle_rows.append(
            {
                "phase_fraction": float(phase_value),
                "theta1_deg": float(angles_deg[0]),
                "theta2_deg": float(angles_deg[1]),
                "theta3_deg": float(angles_deg[2]),
            }
        )
    save_csv(
        ANGLES_FULL_CSV,
        angle_rows,
        ["phase_fraction", "theta1_deg", "theta2_deg", "theta3_deg"],
    )

    save_csv(
        ANGLES_KEY_CSV,
        key_rows,
        ["fase", "phase_fraction", "x_bl", "y_bl", "theta1_deg", "theta2_deg", "theta3_deg"],
    )

    params = np.asarray(fourbar_solution["params"], dtype=float)
    x0, y0, phi, d, a, b, c, px, py, theta_offset = params
    o2 = np.array([x0, y0], dtype=float)
    o4 = o2 + d * np.array([math.cos(phi), math.sin(phi)], dtype=float)
    fourbar_rows = [
        {"parameter": "ground_length_d_bl", "value": float(d)},
        {"parameter": "input_link_a_bl", "value": float(a)},
        {"parameter": "coupler_link_b_bl", "value": float(b)},
        {"parameter": "output_link_c_bl", "value": float(c)},
        {"parameter": "fixed_pivot_O2_x_bl", "value": float(o2[0])},
        {"parameter": "fixed_pivot_O2_y_bl", "value": float(o2[1])},
        {"parameter": "fixed_pivot_O4_x_bl", "value": float(o4[0])},
        {"parameter": "fixed_pivot_O4_y_bl", "value": float(o4[1])},
        {"parameter": "ground_orientation_deg", "value": float(np.rad2deg(phi))},
        {"parameter": "tracing_point_px_bl", "value": float(px)},
        {"parameter": "tracing_point_py_bl", "value": float(py)},
        {"parameter": "coupler_plate_AP_bl", "value": float(np.hypot(px, py))},
        {"parameter": "coupler_plate_BP_bl", "value": float(np.hypot(px - b, py))},
        {"parameter": "motor_pivot", "value": "O2"},
        {"parameter": "input_theta_offset_deg", "value": float(np.rad2deg(theta_offset))},
        {"parameter": "input_rotation_per_cycle_deg", "value": 360.0},
        {"parameter": "rms_error_design_bl", "value": float(fourbar_solution["rms_error_bl"])},
        {"parameter": "rms_error_raw_bl", "value": float(fourbar_solution["rms_error_raw_bl"])},
        {"parameter": "closure_error_bl", "value": float(fourbar_solution["closure_error_bl"])},
        {"parameter": "min_upper_margin_bl", "value": float(fourbar_solution["min_upper_margin_bl"])},
        {"parameter": "min_lower_margin_bl", "value": float(fourbar_solution["min_lower_margin_bl"])},
        {"parameter": "min_projected_clearance_bl", "value": float(fourbar_collision_summary["min_clearance_projected_bl"])},
        {"parameter": "min_same_layer_clearance_bl", "value": float(fourbar_collision_summary["min_clearance_physical_bl"])},
        {"parameter": "max_recommended_link_radius_bl", "value": float(fourbar_collision_summary["max_recommended_link_radius_bl"])},
        {"parameter": "configuration_branch", "value": float(int(fourbar_solution["branch"]))},
    ]
    save_csv(FOURBAR_CSV, fourbar_rows, ["parameter", "value"])
    save_csv(
        FOURBAR_COLLISION_CSV,
        fourbar_collision_rows,
        [
            "phase_fraction",
            "segment_a",
            "segment_b",
            "layer_a",
            "layer_b",
            "same_layer",
            "clearance_bl",
            "projected_collision",
            "physical_collision",
        ],
    )
    save_csv(
        FOURBAR_STATES_CSV,
        fourbar_state_rows,
        [
            "phase_fraction",
            "O2_x_bl",
            "O2_y_bl",
            "A_x_bl",
            "A_y_bl",
            "B_x_bl",
            "B_y_bl",
            "O4_x_bl",
            "O4_y_bl",
            "P_x_bl",
            "P_y_bl",
            "theta2_deg",
            "theta3_deg",
            "theta4_deg",
            "upper_margin_bl",
            "lower_margin_bl",
            "min_projected_clearance_bl",
            "min_same_layer_clearance_bl",
            "projected_collision",
            "physical_collision",
        ],
    )

    overlay_rows = []
    for phase_value, xy_target, xy_fourbar in zip(
        reference_cycle["phase"],
        reference_cycle["xy"],
        np.asarray(fourbar_solution["trace"], dtype=float),
    ):
        design_xy = np.asarray(design_cycle["xy"], dtype=float)
        overlay_rows.append(
            {
                "phase_fraction": float(phase_value),
                "x_raw_target_bl": float(xy_target[0]),
                "y_raw_target_bl": float(xy_target[1]),
                "x_design_target_bl": float(design_xy[len(overlay_rows), 0]),
                "y_design_target_bl": float(design_xy[len(overlay_rows), 1]),
                "x_fourbar_bl": float(xy_fourbar[0]),
                "y_fourbar_bl": float(xy_fourbar[1]),
                "pointwise_error_raw_bl": float(np.linalg.norm(xy_fourbar - xy_target)),
                "pointwise_error_design_bl": float(np.linalg.norm(xy_fourbar - design_xy[len(overlay_rows)])),
            }
        )
    save_csv(
        MECHANISM_TRACE_CSV,
        overlay_rows,
        [
            "phase_fraction",
            "x_raw_target_bl",
            "y_raw_target_bl",
            "x_design_target_bl",
            "y_design_target_bl",
            "x_fourbar_bl",
            "y_fourbar_bl",
            "pointwise_error_raw_bl",
            "pointwise_error_design_bl",
        ],
    )

    mujoco_rows = []
    for phase_value, raw_xy, design_xy, p_xy, a_xy, b_xy in zip(
        mujoco_solution["phase"],
        mujoco_solution["raw_target_xy"],
        mujoco_solution["design_target_xy"],
        mujoco_solution["trace_xy"],
        mujoco_solution["a_trace_xy"],
        mujoco_solution["b_trace_xy"],
    ):
        mujoco_rows.append(
            {
                "phase_fraction": float(phase_value),
                "x_raw_target_bl": float(raw_xy[0]),
                "y_raw_target_bl": float(raw_xy[1]),
                "x_design_target_bl": float(design_xy[0]),
                "y_design_target_bl": float(design_xy[1]),
                "x_mujoco_p_bl": float(p_xy[0]),
                "y_mujoco_p_bl": float(p_xy[1]),
                "x_mujoco_a_bl": float(a_xy[0]),
                "y_mujoco_a_bl": float(a_xy[1]),
                "x_mujoco_b_bl": float(b_xy[0]),
                "y_mujoco_b_bl": float(b_xy[1]),
            }
        )
    save_csv(
        MUJOCO_TRACE_CSV,
        mujoco_rows,
        [
            "phase_fraction",
            "x_raw_target_bl",
            "y_raw_target_bl",
            "x_design_target_bl",
            "y_design_target_bl",
            "x_mujoco_p_bl",
            "y_mujoco_p_bl",
            "x_mujoco_a_bl",
            "y_mujoco_a_bl",
            "x_mujoco_b_bl",
            "y_mujoco_b_bl",
        ],
    )

    summary = {
        "course": COURSE_METADATA,
        "group_members": GROUP_MEMBERS,
        "video_metadata": metadata,
        "mean_body_length_px": analyzed_mean_body_length_px,
        "tracked_frames": DISTAL_TRACK_FRAMES.tolist(),
        "primary_tip_track_bl": np.column_stack(
            [distal_records["primary_tip_x_bl"], distal_records["primary_tip_y_bl"]]
        ).tolist(),
        "secondary_tip_track_bl": np.column_stack(
            [distal_records["secondary_tip_x_bl"], distal_records["secondary_tip_y_bl"]]
        ).tolist(),
        "phase_markers": PHASE_MARKERS,
        "phase_description": PHASE_DESCRIPTION,
        "fourbar_solution": {
            "params": params.tolist(),
            "branch": int(fourbar_solution["branch"]),
            "rms_error_bl": float(fourbar_solution["rms_error_bl"]),
            "rms_error_raw_bl": float(fourbar_solution["rms_error_raw_bl"]),
            "closure_error_bl": float(fourbar_solution["closure_error_bl"]),
            "min_upper_margin_bl": float(fourbar_solution["min_upper_margin_bl"]),
            "min_lower_margin_bl": float(fourbar_solution["min_lower_margin_bl"]),
            "collision_summary": fourbar_collision_summary,
        },
        "outputs": {
            "body_track_csv": str(BODY_TRACK_CSV),
            "distal_track_csv": str(DISTAL_TRACK_CSV),
            "distal_cycle_csv": str(DISTAL_CYCLE_CSV),
            "design_cycle_csv": str(DESIGN_CYCLE_CSV),
            "phase_csv": str(PHASE_CSV),
            "angles_full_csv": str(ANGLES_FULL_CSV),
            "angles_key_csv": str(ANGLES_KEY_CSV),
            "fourbar_csv": str(FOURBAR_CSV),
            "fourbar_collision_csv": str(FOURBAR_COLLISION_CSV),
            "fourbar_states_csv": str(FOURBAR_STATES_CSV),
            "mechanism_trace_csv": str(MECHANISM_TRACE_CSV),
            "mujoco_trace_csv": str(MUJOCO_TRACE_CSV),
            "mujoco_gif": str(MUJOCO_GIF),
            "mujoco_match_figure": str(FIGURE_SPECS["mujoco_match"]),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def run_pipeline() -> dict[str, object]:
    ensure_output_dir()
    copy_inputs()
    frames, fps = read_video_frames(LOCAL_VIDEO)
    metadata = video_metadata(frames, fps)

    body_track = track_body(frames, fps)
    distal_records = build_distal_tracking_records(body_track)
    reference_cycle = build_reference_cycle(distal_records)
    design_cycle = build_design_cycle(reference_cycle, order=DESIGN_FOURIER_ORDER)
    angle_solution = solve_equivalent_angles(reference_cycle)
    key_rows = key_phase_rows(angle_solution, reference_cycle)
    fourbar_solution = synthesize_fourbar(np.asarray(design_cycle["xy"], dtype=float), np.asarray(design_cycle["phase"], dtype=float))
    fourbar_solution["rms_error_raw_bl"] = float(
        np.sqrt(np.mean(np.sum((np.asarray(fourbar_solution["trace"], dtype=float) - reference_cycle["xy"]) ** 2, axis=1)))
    )
    fourbar_collision_rows, fourbar_collision_summary = evaluate_fourbar_collisions(fourbar_solution)
    fourbar_state_rows = build_fourbar_state_rows(fourbar_solution, np.asarray(reference_cycle["phase"], dtype=float))
    mujoco_solution = build_mujoco_gif(reference_cycle, design_cycle, fourbar_solution)

    create_video_sheet(frames)
    create_tracked_points_keyframes(frames, distal_records)
    create_body_path_overlay(frames, body_track, distal_records)
    create_ciclogram(reference_cycle, design_cycle, distal_records)
    create_phase_plot()
    create_joint_angle_plot(angle_solution)
    create_overlay_plot(reference_cycle, design_cycle, fourbar_solution)
    create_fourbar_full_cycle_plot(reference_cycle, design_cycle, fourbar_solution, fourbar_collision_summary)
    create_fourbar_clearance_plot(fourbar_state_rows)
    create_fourbar_cable_tpu_concept_plot(fourbar_solution)
    create_mechanism_snapshots(fourbar_solution, reference_cycle)

    write_outputs(
        metadata,
        body_track,
        distal_records,
        reference_cycle,
        design_cycle,
        angle_solution,
        key_rows,
        fourbar_solution,
        fourbar_collision_rows,
        fourbar_collision_summary,
        fourbar_state_rows,
        mujoco_solution,
    )

    return {
        "output_dir": str(OUTPUT_DIR),
        "video_metadata": metadata,
        "mean_body_length_px": float(
            np.mean(body_track["body_length_px"][: DISTAL_TRACK_FRAMES[-1] + 1])
        ),
        "design_cycle_order": int(design_cycle["order"]),
        "design_cycle_rms_to_reference_bl": float(design_cycle["rms_to_reference_bl"]),
        "tracked_frames": DISTAL_TRACK_FRAMES.tolist(),
        "key_angles": key_rows,
        "fourbar_rms_error_design_bl": float(fourbar_solution["rms_error_bl"]),
        "fourbar_rms_error_raw_bl": float(fourbar_solution["rms_error_raw_bl"]),
        "fourbar_collision_summary": fourbar_collision_summary,
        "csv_outputs": {
            "body_track": str(BODY_TRACK_CSV),
            "distal_track": str(DISTAL_TRACK_CSV),
            "trajectory_raw": str(DISTAL_CYCLE_CSV),
            "trajectory_design": str(DESIGN_CYCLE_CSV),
            "phase": str(PHASE_CSV),
            "angles_full": str(ANGLES_FULL_CSV),
            "angles_key": str(ANGLES_KEY_CSV),
            "fourbar": str(FOURBAR_CSV),
            "fourbar_states": str(FOURBAR_STATES_CSV),
            "fourbar_collision": str(FOURBAR_COLLISION_CSV),
            "overlay": str(MECHANISM_TRACE_CSV),
            "mujoco": str(MUJOCO_TRACE_CSV),
        },
        "mujoco_gif": str(MUJOCO_GIF),
    }


if __name__ == "__main__":
    summary = run_pipeline()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
