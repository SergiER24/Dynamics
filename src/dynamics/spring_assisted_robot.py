"""Generate portable robotics evidence for the spring-assisted mechanism portfolio."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

LINK_1_M = 0.70
LINK_2_M = 0.70
BASE_HEIGHT_M = 0.05
TRAJECTORY_DURATION_S = 6.90
ACTIVE_CURRENT_THRESHOLD_MA = 10.0

WITHOUT_SPRING_CSV = DATA / "Grupo5_Sin_Resorte(in).csv"
WITH_SPRING_CSV = DATA / "Grupo5_Con_Resorte(in).csv"


def forward_kinematics(q1_rad: np.ndarray | float, q2_rad: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Return end-effector x and z coordinates for the planar RR reference model."""
    q1 = np.asarray(q1_rad)
    q2 = np.asarray(q2_rad)
    x_m = LINK_1_M * np.cos(q1) + LINK_2_M * np.cos(q1 + q2)
    z_m = BASE_HEIGHT_M - LINK_1_M * np.sin(q1) - LINK_2_M * np.sin(q1 + q2)
    return x_m, z_m


def jacobian(q1_rad: float, q2_rad: float) -> np.ndarray:
    """Return the planar RR geometric Jacobian."""
    return np.array(
        [
            [
                -LINK_1_M * np.sin(q1_rad) - LINK_2_M * np.sin(q1_rad + q2_rad),
                -LINK_2_M * np.sin(q1_rad + q2_rad),
            ],
            [
                -LINK_1_M * np.cos(q1_rad) - LINK_2_M * np.cos(q1_rad + q2_rad),
                -LINK_2_M * np.cos(q1_rad + q2_rad),
            ],
        ]
    )


def quintic_scale(time_s: np.ndarray, duration_s: float = TRAJECTORY_DURATION_S) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized rest-to-rest position, velocity, and acceleration scales."""
    tau = np.clip(np.asarray(time_s, dtype=float) / duration_s, 0.0, 1.0)
    position = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    velocity = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / duration_s
    acceleration = (60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3) / duration_s**2
    return position, velocity, acceleration


def reference_trajectory() -> dict[str, np.ndarray]:
    """Generate a smooth reference trajectory for the portable RR example."""
    time_s = np.linspace(0.0, TRAJECTORY_DURATION_S, 500)
    scale, velocity_scale, acceleration_scale = quintic_scale(time_s)
    q_start = np.array([0.0, 0.0])
    q_goal = np.array([-0.8, 0.8])
    delta = q_goal - q_start
    q = q_start + np.outer(scale, delta)
    qd = np.outer(velocity_scale, delta)
    qdd = np.outer(acceleration_scale, delta)
    x_m, z_m = forward_kinematics(q[:, 0], q[:, 1])
    condition_number = np.array([np.linalg.cond(jacobian(q1, q2)) for q1, q2 in q])
    return {
        "time_s": time_s,
        "q_rad": q,
        "qd_rad_s": qd,
        "qdd_rad_s2": qdd,
        "x_m": x_m,
        "z_m": z_m,
        "jacobian_condition_number": condition_number,
    }


def workspace_points(n_samples: int = 220) -> tuple[np.ndarray, np.ndarray]:
    """Return a dense workspace sample for the planar RR reference model."""
    q1 = np.linspace(-np.pi, np.pi, n_samples)
    q2 = np.linspace(-np.pi, np.pi, n_samples)
    q1_grid, q2_grid = np.meshgrid(q1, q2)
    return forward_kinematics(q1_grid, q2_grid)


def load_current_measurement(path: Path) -> pd.DataFrame:
    """Load and normalize a servomotor-current measurement trace."""
    frame = pd.read_csv(path).rename(
        columns={
            "Tiempo (s)": "time_s",
            "Corriente Servo 1 (mA)": "servo_1_mA",
            "Corriente Servo 2 (mA)": "servo_2_mA",
        }
    )
    frame = frame.sort_values("time_s").reset_index(drop=True)
    frame["time_s"] -= float(frame["time_s"].iloc[0])
    return frame


def active_window(frame: pd.DataFrame, threshold_ma: float = ACTIVE_CURRENT_THRESHOLD_MA) -> pd.DataFrame:
    """Trim a trace to the active interval identified by total absolute current."""
    total = frame[["servo_1_mA", "servo_2_mA"]].abs().sum(axis=1)
    active_indices = np.flatnonzero(total.to_numpy() > threshold_ma)
    if len(active_indices) == 0:
        return frame.copy()
    trimmed = frame.iloc[int(active_indices[0]) : int(active_indices[-1]) + 1].copy().reset_index(drop=True)
    trimmed["time_s"] -= float(trimmed["time_s"].iloc[0])
    return trimmed


def current_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return reviewable current metrics for both servomotors."""
    metrics: dict[str, dict[str, float]] = {}
    time_s = frame["time_s"].to_numpy(dtype=float)
    for servo, column in [("servo_1", "servo_1_mA"), ("servo_2", "servo_2_mA")]:
        values = frame[column].to_numpy(dtype=float)
        metrics[servo] = {
            "mean_absolute_current_ma": float(np.mean(np.abs(values))),
            "rms_current_ma": float(np.sqrt(np.mean(values**2))),
            "peak_absolute_current_ma": float(np.max(np.abs(values))),
            "absolute_charge_mas": float(np.trapezoid(np.abs(values), time_s)),
        }
    return metrics


def _percent_change(without_value: float, with_value: float) -> float:
    return 100.0 * (with_value - without_value) / without_value


def analyze_current_measurements() -> dict[str, object]:
    """Compare active-window current traces with and without spring assistance."""
    without_spring = active_window(load_current_measurement(WITHOUT_SPRING_CSV))
    with_spring = active_window(load_current_measurement(WITH_SPRING_CSV))
    without_metrics = current_metrics(without_spring)
    with_metrics = current_metrics(with_spring)
    changes: dict[str, dict[str, float]] = {}
    for servo in ["servo_1", "servo_2"]:
        changes[servo] = {
            f"{key}_change_percent": _percent_change(without_metrics[servo][key], with_metrics[servo][key])
            for key in without_metrics[servo]
        }
    return {
        "without_spring": without_metrics,
        "with_spring": with_metrics,
        "change_percent": changes,
        "without_spring_trace": without_spring,
        "with_spring_trace": with_spring,
    }


def _configure_plotting() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"savefig.dpi": 220, "axes.spines.top": False, "axes.spines.right": False})


def _save_workspace_plot(trajectory: dict[str, np.ndarray]) -> None:
    workspace_x, workspace_z = workspace_points()
    figure, axis = plt.subplots(figsize=(7.2, 6.0))
    axis.scatter(workspace_x[::3, ::3], workspace_z[::3, ::3], s=1.0, alpha=0.10, label="Reachable workspace")
    axis.plot(trajectory["x_m"], trajectory["z_m"], color="#b6422c", linewidth=2.2, label="Reference path")
    axis.scatter([trajectory["x_m"][0], trajectory["x_m"][-1]], [trajectory["z_m"][0], trajectory["z_m"][-1]], color=["#146b55", "#6b3fa0"], s=55, label="Boundary configurations")
    axis.set_aspect("equal")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("z [m]")
    axis.set_title("Planar RR workspace and smooth reference trajectory")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(FIGURES / "rr_workspace_and_path.png")
    plt.close(figure)


def _save_jacobian_plot(trajectory: dict[str, np.ndarray]) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    axis.semilogy(trajectory["time_s"], trajectory["jacobian_condition_number"], linewidth=2.0)
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Jacobian condition number")
    axis.set_title("Kinematic conditioning along the RR reference trajectory")
    figure.tight_layout()
    figure.savefig(FIGURES / "jacobian_condition_number.png")
    plt.close(figure)


def _save_current_plot(measurements: dict[str, object]) -> None:
    without_spring = measurements["without_spring_trace"]
    with_spring = measurements["with_spring_trace"]
    figure, axes = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=False)
    for axis, column, title in [
        (axes[0], "servo_1_mA", "Servo 1 current"),
        (axes[1], "servo_2_mA", "Servo 2 current"),
    ]:
        axis.plot(without_spring["time_s"], without_spring[column], label="Without spring", color="#b6422c")
        axis.plot(with_spring["time_s"], with_spring[column], label="With spring", color="#146b55")
        axis.set_ylabel("Current [mA]")
        axis.set_title(title)
        axis.legend()
    axes[1].set_xlabel("Active-window time [s]")
    figure.suptitle("Experimental servomotor-current comparison")
    figure.tight_layout()
    figure.savefig(FIGURES / "servo_current_comparison.png")
    plt.close(figure)


def write_artifacts() -> dict[str, object]:
    """Generate portfolio figures and a machine-readable summary."""
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    _configure_plotting()
    trajectory = reference_trajectory()
    measurements = analyze_current_measurements()
    _save_workspace_plot(trajectory)
    _save_jacobian_plot(trajectory)
    _save_current_plot(measurements)
    summary = {
        "project": "Spring-Assisted Robotic Mechanism",
        "portable_rr_reference": {
            "link_1_m": LINK_1_M,
            "link_2_m": LINK_2_M,
            "trajectory_duration_s": TRAJECTORY_DURATION_S,
            "start_position_m": [float(trajectory["x_m"][0]), float(trajectory["z_m"][0])],
            "final_position_m": [float(trajectory["x_m"][-1]), float(trajectory["z_m"][-1])],
            "maximum_finite_jacobian_condition_number": float(np.max(trajectory["jacobian_condition_number"][np.isfinite(trajectory["jacobian_condition_number"])])),
        },
        "experimental_current_analysis": {
            "without_spring": measurements["without_spring"],
            "with_spring": measurements["with_spring"],
            "change_percent": measurements["change_percent"],
        },
        "workshop_reported_spring_design": {
            "attachment_point": "C",
            "stiffness_n_m": 4.0,
            "free_length_mm": 53.72,
            "maximum_force_n": 0.309,
            "peak_torque_reduction_percent": {"motor_1": 53.59, "motor_2": 68.95},
            "rms_torque_reduction_percent": {"motor_1": 45.02, "motor_2": 85.61},
        },
    }
    (RESULTS / "dynamics_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = write_artifacts()
    print(json.dumps(summary["experimental_current_analysis"]["change_percent"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
