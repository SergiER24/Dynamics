"""Regression tests for the portable robotics analysis."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dynamics"))

import spring_assisted_robot as model


class SpringAssistedRobotTests(unittest.TestCase):
    def test_forward_kinematics_at_zero_configuration(self) -> None:
        x_m, z_m = model.forward_kinematics(0.0, 0.0)
        self.assertAlmostEqual(float(x_m), model.LINK_1_M + model.LINK_2_M)
        self.assertAlmostEqual(float(z_m), model.BASE_HEIGHT_M)

    def test_quintic_boundary_conditions(self) -> None:
        position, velocity, acceleration = model.quintic_scale(np.array([0.0, model.TRAJECTORY_DURATION_S]))
        np.testing.assert_allclose(position, [0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(velocity, [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(acceleration, [0.0, 0.0], atol=1e-12)

    def test_current_analysis_is_finite(self) -> None:
        analysis = model.analyze_current_measurements()
        for case in ["without_spring", "with_spring"]:
            for servo_metrics in analysis[case].values():
                self.assertTrue(all(np.isfinite(value) for value in servo_metrics.values()))


if __name__ == "__main__":
    unittest.main()
