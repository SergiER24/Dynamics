# Spring-Assisted Robotic Mechanism

This GitHub Pages-ready site documents the featured project from the Dynamics
CPO at Universidad de los Andes.

## Engineering Workflow

```mermaid
flowchart LR
    A[Mechanism geometry] --> B[Forward kinematics]
    B --> C[Workspace analysis]
    C --> D[Inverse kinematics]
    D --> E[Piecewise quintic trajectory]
    E --> F[Inverse dynamics]
    F --> G[Spring configuration search]
    G --> H[MuJoCo visualization]
    H --> I[Experimental current comparison]
```

## Documentation Map

- [Mathematical formulation](mathematical-formulation.md)
- [Reproducibility guide](reproducibility.md)
- [Portfolio evaluation](portfolio-evaluation.md)

## Generated Evidence

![Workspace](assets/rr_workspace_and_path.png)

![Current comparison](assets/servo_current_comparison.png)
