# Mathematical Formulation

## RR Reference Kinematics

For the planar two-link reference arm,

```math
x = L_1\cos q_1 + L_2\cos(q_1+q_2)
```

```math
z = z_0 - L_1\sin q_1 - L_2\sin(q_1+q_2).
```

The geometric Jacobian is

```math
J(q)=
\begin{bmatrix}
-L_1\sin q_1-L_2\sin(q_1+q_2) & -L_2\sin(q_1+q_2)\\
-L_1\cos q_1-L_2\cos(q_1+q_2) & -L_2\cos(q_1+q_2)
\end{bmatrix}.
```

## Rest-to-Rest Quintic Trajectory

With normalized time `\tau=t/T`,

```math
s(\tau)=10\tau^3-15\tau^4+6\tau^5
```

```math
q(t)=q_i+(q_f-q_i)s(\tau).
```

This interpolation enforces zero velocity and acceleration at both boundaries.
The full workshop extends the method to piecewise splines across task
waypoints.

## Inverse Dynamics

The actuator torque structure is

```math
\tau=M(q)\ddot q+C(q,\dot q)\dot q+g(q)+\tau_f-\tau_s.
```

The workshop evaluates this expression over the payload trajectory and then
adds a tensile linear spring:

```math
F_s=k\max(L-L_0,0).
```

## Spring-Design Objective

The design search considers attachment point, fixed anchor, free length, and
stiffness. A practical objective penalizes both peak and RMS torque:

```math
\min_\theta
\sum_i
\left(
w_p \max_t |\tau_i(t;\theta)|
+
w_r \sqrt{\frac{1}{T}\int_0^T \tau_i^2(t;\theta)\,dt}
\right).
```

## Experimental Validation

Measured servomotor current is analyzed as a practical proxy for actuator
demand. The portable workflow trims inactive periods and reports mean absolute
current, RMS current, peak current, and integrated absolute current. These
measurements do not uniformly confirm the theoretical torque reduction, so the
repository reports them as an experimental comparison rather than a completed
validation claim.
