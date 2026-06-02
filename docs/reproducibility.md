# Reproducibility Guide

## Portable Workflow

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/dynamics/spring_assisted_robot.py
python -m unittest discover -s tests -v
```

Generated artifacts:

- `figures/rr_workspace_and_path.png`
- `figures/jacobian_condition_number.png`
- `figures/servo_current_comparison.png`
- `results/dynamics_summary.json`

## Advanced Coursework Workflows

Install the optional packages:

```bash
python -m pip install -r requirements-mujoco.txt
```

Supporting MuJoCo, mechanism-synthesis, and notebook workflows are preserved
under `src/coursework/` and `notebooks/coursework/`. Some advanced workflows
require external course inputs that are intentionally not versioned.
