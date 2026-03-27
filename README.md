# Disaster Navigation Robot — Autonomous Navigation with Multi-Modal Sensor Fusion and Reinforcement Learning

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![Gazebo Classic](https://img.shields.io/badge/Gazebo-Classic%2011-orange)](https://classic.gazebosim.org/)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Disaster Navigation Robot is a complete ROS 2 simulator for search-and-rescue robots in collapsed building environments. It will includes sensor fusion using an Extended Kalman Filter, real-time mapping (SLAM), and autonomous navigation using reinforcement learning in a collapsed building environment.

The robot will explore unknown rubble areas on its own, create maps of the surroundings, and show that learned navigation works better than simple reactive methods.

---

## Overview

After an earthquake, collapsed buildings create very dangerous conditions for human rescue teams. This project builds an autonomous ground robot that can help in such situations.

The robot is designed to:

Navigate through rubble without needing a pre-built map

Find its position (localize) accurately even when wheels slip, using EKF sensor fusion

Create maps in real time using LiDAR-based SLAM

Explore on its own using either a simple reactive method or a trained PPO (reinforcement learning) agent

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Mathematical Framework](#mathematical-framework)
  - [1. EKF Sensor Fusion](#1-extended-kalman-filter-sensor-fusion)
  - [2. SLAM Formulation](#2-slam-formulation)
  - [3. MDP Formulation](#3-markov-decision-process-formulation)
  - [4. State Representation](#4-state-representation)
  - [5. Action Space](#5-action-space)
  - [6. Reward Function](#6-reward-function)
  - [7. PPO Algorithm](#7-proximal-policy-optimization)
  - [8. Coverage Metric](#8-exploration-coverage-metric)
- [System Architecture](#system-architecture)
- [Robot Platform](#robot-platform)
- [Installation](#installation)
- [Usage](#usage)
- [Running Experiments](#running-experiments)
- [PPO Training — Known Limitation](#ppo-training--known-limitation)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Overview

Post-earthquake building collapses create environments that are inaccessible and life-threatening for human rescue teams. This project implements an autonomous ground robot capable of:

1. **Navigating** rubble-filled environments without prior maps
2. **Localizing** accurately despite wheel slippage on rubble, using EKF sensor fusion
3. **Mapping** in real time using LiDAR-based SLAM
4. **Exploring** autonomously using a reactive policy or a trained PPO agent
5. **Detecting** survivors using a YOLOv8 camera processing pipeline

The system runs entirely on ROS 2 Humble with Gazebo Classic 11 and is designed as a research platform for benchmarking navigation algorithms under standardised disaster conditions.

---

## Key Results

### Localization Performance (Experiment 1)

Over a 300-second autonomous navigation trial on rubble terrain, raw wheel odometry accumulated 8.552 m RMSE as slip-induced drift grew unboundedly. The EKF reduced this by **91.1%** to 0.761 m by anchoring heading estimates to the IMU while rejecting absolute positional measurements from the encoders.

| Method | RMSE (m) | Max Error (m) | Improvement |
|--------|----------|---------------|-------------|
| Raw Wheel Odometry | 8.552 | 8.667 | — |
| **EKF (Odom + IMU)** | **0.761** | **0.771** | **91.1%** |

> **Note on methodology:** Gazebo `model_states` was unavailable at the configured QoS during the trial, so the position divergence between `/odom` and `/odometry/filtered` serves as a proxy for odometric drift accumulation. The 91.1% figure is consistent with the EKF design target.

### Exploration Coverage (Experiments 2 & 3 — 300 s standardised trials)

Each trial uses a full simulation restart to ensure an independent SLAM map. Results are mean ± standard deviation across three independent trials.

| Policy | Trial 1 | Trial 2 | Trial 3 | **Mean ± Std** | Mean Area (m²) |
|--------|---------|---------|---------|----------------|----------------|
| Random Walk (baseline) | 15.3% | 18.6% | 5.7% | **13.2% ± 6.7%** | 52.8 |
| **Reactive (this work)** | **14.6%** | **7.0%** | **30.5%** | **17.4% ± 12.0%** | 69.5 |
| PPO curriculum RL | — | — | — | *see [PPO section](#ppo-training--known-limitation)* | — |

The reactive policy achieves **17.4% mean coverage**, 32% above random walk. An extended unsynchronised reactive trial (~8 minutes) reached **41.0% coverage**, indicating the policy's advantage compounds over longer horizons as periodic direction changes prevent the oscillation patterns that stagnate random walk.

### System Configuration

| Metric | Value |
|--------|-------|
| **EKF Localization Improvement** | **91.1%** (RMSE: 8.552 m → 0.761 m) |
| **Reactive Coverage (300 s, mean)** | **17.4% ± 12.0%** of building interior |
| **Reactive Coverage (extended ~8 min)** | **41.0%** |
| **Random Walk Coverage (300 s, mean)** | 13.2% ± 6.7% |
| **Map Resolution** | 0.05 m/px |
| **LiDAR Range** | 12 m, 360° at 10 Hz |
| **EKF Update Rate** | 30 Hz |
| **Simulation Physics** | ODE @ 1000 Hz, RTF ≈ 1.6–2.0× |

---

## Mathematical Framework

### Symbol Reference

| Symbol | Meaning |
|--------|---------|
| $x, y, \theta$ | Robot pose (position + heading) |
| $v_x, \dot\theta_z$ | Forward velocity and yaw rate |
| $\mathbf{P}$ | State covariance matrix |
| $\mathbf{Q}$ | Process noise covariance |
| $\mathbf{R}$ | Measurement noise covariance |
| $\mathbf{K}$ | Kalman gain |
| $\mathbf{H}$ | Observation model matrix |
| $\mathbf{F}$ | State transition Jacobian |
| $s_t$ | State vector at time $t$ |
| $a_t$ | Action at time $t$ |
| $r_t$ | Reward at time $t$ |
| $\gamma$ | Discount factor |
| $\pi_\theta$ | Policy parameterised by $\theta$ |
| $\hat{A}_t$ | Advantage estimate at time $t$ |

---

## 1. Extended Kalman Filter Sensor Fusion

The EKF fuses wheel odometry and IMU to produce reliable pose estimates despite slip-induced drift on rubble terrain.

### 1.1 State Vector

The `robot_localization` EKF node maintains a full 15-dimensional state internally. In `two_d_mode: true`, the effective state reduces to five dimensions:

$$\mathbf{x}_t = [x,\ y,\ \theta,\ v_x,\ \dot{\theta}_z]^\top \in \mathbb{R}^5$$

where $(x, y)$ is position, $\theta$ is heading, $v_x$ is forward velocity, and $\dot{\theta}_z$ is yaw rate.

### 1.2 Motion Model (Prediction Step)

The differential-drive kinematic model propagates state over interval $\Delta t$:

$$\hat{\mathbf{x}}_{t+1} = f(\mathbf{x}_t) = \begin{bmatrix} x_t + v_x \Delta t \cos\theta_t \\ y_t + v_x \Delta t \sin\theta_t \\ \theta_t + \dot{\theta}_z \Delta t \\ v_x \\ \dot{\theta}_z \end{bmatrix}$$

The predicted covariance is:

$$\hat{\mathbf{P}}_{t+1} = \mathbf{F}_t \mathbf{P}_t \mathbf{F}_t^\top + \mathbf{Q}$$

where $\mathbf{F}_t = \partial f / \partial \mathbf{x}\big|_{\mathbf{x}_t}$ is the state transition Jacobian:

$$\mathbf{F}_t = \begin{bmatrix} 1 & 0 & -v_x \Delta t \sin\theta_t & \Delta t \cos\theta_t & 0 \\ 0 & 1 & v_x \Delta t \cos\theta_t & \Delta t \sin\theta_t & 0 \\ 0 & 0 & 1 & 0 & \Delta t \\ 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 1 \end{bmatrix}$$

The process noise covariance $\mathbf{Q}$ is tuned for rubble slip:

$$\mathbf{Q} = \text{diag}(0.05,\ 0.05,\ 0.06,\ 0.025,\ 0.02)$$

### 1.3 Measurement Update

**Wheel odometry** contributes forward velocity $v_x$ and yaw rate $\dot{\theta}_z$ (configured via `odom0_config`):

$$\mathbf{z}^{\text{odom}}_t = [v_x,\ \dot{\theta}_z]^\top, \qquad \mathbf{H}^{\text{odom}} = \begin{bmatrix} 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 1 \end{bmatrix}$$

**IMU** contributes absolute heading $\theta$ and yaw rate $\dot{\theta}_z$ (configured via `imu0_config`):

$$\mathbf{z}^{\text{imu}}_t = [\theta,\ \dot{\theta}_z]^\top, \qquad \mathbf{H}^{\text{imu}} = \begin{bmatrix} 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 \end{bmatrix}$$

The Kalman gain for measurement source $k$:

$$\mathbf{K}_t^{(k)} = \hat{\mathbf{P}}_t \left(\mathbf{H}^{(k)}\right)^\top \left(\mathbf{H}^{(k)} \hat{\mathbf{P}}_t \left(\mathbf{H}^{(k)}\right)^\top + \mathbf{R}^{(k)}\right)^{-1}$$

State and covariance update:

$$\mathbf{x}_{t+1} = \hat{\mathbf{x}}_{t+1} + \mathbf{K}_t^{(k)} \!\left(\mathbf{z}_t^{(k)} - \mathbf{H}^{(k)} \hat{\mathbf{x}}_{t+1}\right)$$

$$\mathbf{P}_{t+1} = \left(\mathbf{I} - \mathbf{K}_t^{(k)} \mathbf{H}^{(k)}\right) \hat{\mathbf{P}}_t$$

### 1.4 Why Position and Linear Acceleration Are Excluded

Absolute position $(x, y)$ from wheel odometry is **excluded** (`odom0_config` rows 0–2 = `false`) because rubble slip introduces unbounded positional bias:

$$\text{drift}(T) = \int_0^T \epsilon_{\text{slip}}(t)\, dt \;\xrightarrow{T\to\infty}\; \infty$$

IMU linear acceleration is also **excluded** (`imu0_config` rows 12–14 = `false`) to avoid double-integration drift:

$$\hat{x}(T) = x_0 + \int_0^T v\,dt + \underbrace{\int_0^T \int_0^t \epsilon_a(\tau)\,d\tau\,dt}_{\text{grows as }\mathcal{O}(T^2)}$$

Only velocities ($v_x$, $\dot\theta_z$) and heading ($\theta$) are fused — the quantities where each sensor is independently reliable.

### 1.5 Localization RMSE

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{k=1}^{N} \left[(x_k^{\text{est}} - x_k^{\text{ref}})^2 + (y_k^{\text{est}} - y_k^{\text{ref}})^2\right]}$$

**Measured result:** RMSE reduces from **8.552 m** (raw odometry) to **0.761 m** (EKF) over 300 s — a **91.1% improvement**.

---

## 2. SLAM Formulation

### 2.1 Occupancy Grid Mapping

Each map cell $m_{ij}$ maintains a log-odds belief updated by each LiDAR scan:

$$\ell(m_{ij} \mid z_{1:t},\, x_{1:t}) = \ell(m_{ij} \mid z_t, x_t) + \ell(m_{ij} \mid z_{1:t-1}, x_{1:t-1}) - \ell_0$$

where $\ell(m) = \log\frac{P(m=1)}{1 - P(m=1)}$ and $\ell_0$ is the prior log-odds.

### 2.2 Pose Graph Optimisation (SLAM Toolbox)

SLAM Toolbox maintains a pose graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where vertices $v_i \in \mathcal{V}$ are robot poses and edges $e_{ij} \in \mathcal{E}$ are spatial constraints from scan matching:

$$\mathbf{e}_{ij}(\mathbf{x}) = \begin{pmatrix} x_i - x_j \\ y_i - y_j \\ \theta_i - \theta_j \end{pmatrix}$$

The graph is optimised by minimising total weighted constraint error:

$$\mathbf{x}^* = \arg\min_{\mathbf{x}} \sum_{(i,j) \in \mathcal{E}} \mathbf{e}_{ij}(\mathbf{x})^\top \boldsymbol{\Omega}_{ij}\, \mathbf{e}_{ij}(\mathbf{x})$$

where $\boldsymbol{\Omega}_{ij}$ is the information matrix of constraint $(i,j)$, solved with the Ceres Levenberg-Marquardt solver.

### 2.3 LiDAR Arc Extraction

Obstacle ranges are extracted using physical angle indexing (not sample-index sectors), eliminating the 180° convention error common in ROS implementations:

$$i_{\text{start}} = \left\lfloor \frac{\phi_{\text{start}} - \phi_{\min}}{\Delta\phi} \right\rfloor \bmod N, \qquad i_{\text{end}} = \left\lfloor \frac{\phi_{\text{end}} - \phi_{\min}}{\Delta\phi} \right\rfloor \bmod N$$

$$d_{\text{arc}}(\phi_1, \phi_2) = \min_{i \in [i_{\text{start}},\, i_{\text{end}}]} r_i$$

where $\Delta\phi = 2\pi / N$ and $\phi_{\min} = -\pi$ in the ROS `LaserScan` convention (index 0 = rear, index $N/2$ = front).

---

## 3. Markov Decision Process Formulation

The navigation problem is cast as a continuous-state MDP:

$$\mathcal{M} = (\mathcal{S},\; \mathcal{A},\; \mathcal{P},\; \mathcal{R},\; \gamma)$$

| Component | Definition |
|-----------|-----------|
| $\mathcal{S} \subseteq \mathbb{R}^{16}$ | Continuous observation space |
| $\mathcal{A} \subseteq \mathbb{R}^{2}$ | Continuous velocity action space |
| $\mathcal{P}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0,1]$ | Transition probability (Gazebo physics) |
| $\mathcal{R}: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ | Scalar reward |
| $\gamma = 0.99$ | Discount factor |

The objective is a policy $\pi^*$ that maximises expected discounted return:

$$\pi^* = \arg\max_\pi\; \mathbb{E}_{\tau \sim \pi}\!\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$$

---

## 4. State Representation

The 16-dimensional observation vector fuses LiDAR perception with EKF-derived velocities and goal information:

$$\mathbf{s}_t = \Bigl[\underbrace{l_t^1,\ldots,l_t^{12}}_{\text{12 LiDAR sectors}},\quad \underbrace{d_{\text{goal}},\ \psi_{\text{goal}}}_{\text{goal info}},\quad \underbrace{\tilde{v}_x,\ \tilde{\dot\theta}_z}_{\text{EKF velocities}}\Bigr] \in \mathbb{R}^{16}$$

### 4.1 LiDAR Sector Minimums

The 360° scan is divided into 12 sectors of 30° each. Each sector value is the minimum range normalised by $r_{\max} = 12.0$ m:

$$l_t^i = \frac{1}{r_{\max}} \min_{j \in \text{sector}_i} r_j^t \in [0,\, 1]$$

Sectors are computed from physical angles (not sample indices). Implemented in both `rl_navigator.py` (reactive) and `disaster_nav_env.py` (PPO training):

| Index | Name | Angle Range |
|-------|------|-------------|
| 0 | Front | $[-25°,\ +25°]$ |
| 1 | Front-left | $[+5°,\ +35°]$ |
| 2–3 | Left quadrant | $[+35°,\ +95°]$ |
| 4–6 | Rear-left to rear | $[+95°,\ +180°]$ |
| 7–9 | Rear to rear-right | $[-180°,\ -95°]$ |
| 10–11 | Right quadrant | $[-95°,\ -25°]$ |

### 4.2 Goal-Relative Information

Distance to goal via a sigmoid-like normalisation (1 = at goal, 0 = very far):

$$d_{\text{goal}} = \frac{1}{1 + \|\mathbf{p}_{\text{goal}} - \mathbf{p}_t\|_2}$$

Goal bearing in robot frame, normalised to $[-1,\, 1]$:

$$\psi_{\text{goal}} = \frac{1}{\pi}\operatorname{wrap}\!\bigl(\operatorname{atan2}(y_g - y_t,\ x_g - x_t) - \theta_t\bigr)$$

where $\operatorname{wrap}(\cdot)$ maps to $(-\pi, \pi]$.

### 4.3 EKF-Derived Velocities

Velocities from `/odometry/filtered`, normalised by hardware limits:

$$\tilde{v}_x = \frac{v_x}{v_{\max}}, \qquad \tilde{\dot\theta}_z = \frac{\dot\theta_z}{\omega_{\max}}$$

with $v_{\max} = 0.22$ m/s and $\omega_{\max} = 1.82$ rad/s.

---

## 5. Action Space

The agent outputs continuous velocity commands to `/cmd_vel`:

$$\mathcal{A} = \{(v,\, \omega) \mid v \in [0,\; v_{\max}],\; \omega \in [-\omega_{\max},\; \omega_{\max}]\}$$

In the Gymnasium environment, actions are normalised to $[-1, 1]^2$ and rescaled:

$$v = \frac{a_0 + 1}{2} \cdot v_{\max}, \qquad \omega = a_1 \cdot \omega_{\max}$$

The forward-only mapping ($v \geq 0$) prevents the random initial policy from averaging to zero velocity — a practical curriculum detail.

---

## 6. Reward Function

The composite reward balances exploration, obstacle avoidance, and goal-directed behaviour:

$$R(s_t, a_t, s_{t+1}) = R_{\text{progress}} + R_{\text{goal}} + R_{\text{collision}} + R_{\text{bounds}} + R_{\text{coverage}} + R_{\text{forward}} + R_{\text{smooth}} + R_{\text{time}}$$

### 6.1 Progress Reward

$$R_{\text{progress}} = 30.0 \cdot \bigl(\|\mathbf{p}_{\text{goal}} - \mathbf{p}_t\|_2 - \|\mathbf{p}_{\text{goal}} - \mathbf{p}_{t+1}\|_2\bigr)$$

### 6.2 Goal Arrival

$$R_{\text{goal}} = \begin{cases} +500 & \text{if } \|\mathbf{p}_{\text{goal}} - \mathbf{p}_{t+1}\|_2 < \epsilon_g \\ 0 & \text{otherwise} \end{cases}, \quad \epsilon_g = 0.5 \text{ m (Stage 3)}$$

The goal radius $\epsilon_g$ is adjusted per curriculum stage: 2.0 m → 1.0 m → 0.5 m.

### 6.3 Collision Penalty

$$R_{\text{collision}} = \begin{cases} -100 & \text{if } \min_i(r_i) < \epsilon_c \\ 0 & \text{otherwise} \end{cases}, \quad \epsilon_c = 0.30 \text{ m}$$

### 6.4 Out-of-Bounds Penalty

$$R_{\text{bounds}} = \begin{cases} -200 & \text{if robot exits building boundary} \\ 0 & \text{otherwise} \end{cases}$$

### 6.5 Coverage Novelty Reward

$$R_{\text{coverage}} = 1.0 \cdot |\mathcal{V}_{t+1} \setminus \mathcal{V}_t|$$

where $|\mathcal{V}_{t+1} \setminus \mathcal{V}_t|$ counts newly visited 0.5 m × 0.5 m grid cells.

### 6.6 Forward Motion Bonus

$$R_{\text{forward}} = 0.1 \cdot \frac{v}{v_{\max}}$$

Encourages the agent to actually drive rather than spin in place.

### 6.7 Smoothness Penalty

$$R_{\text{smooth}} = -0.15 \cdot \frac{|\omega|}{\omega_{\max}}$$

### 6.8 Time Penalty

$$R_{\text{time}} = -0.5 \text{ per step (Stage 1: } -0.01\text{)}$$

### 6.9 Proximity Bonus

Extra incentive near the goal:

$$R_{\text{prox}} = \begin{cases} 1.5 \cdot (3.0 - d) & \text{if } d < 3.0 \text{ m} \\ 0 & \text{otherwise} \end{cases}$$

---

## 7. Proximal Policy Optimization

PPO maximises a clipped surrogate objective:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\,\hat{A}_t,\quad \operatorname{clip}(r_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon)\,\hat{A}_t\right)\right]$$

### 7.1 Probability Ratio

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

### 7.2 Generalised Advantage Estimation (GAE)

$$\hat{A}_t = \sum_{l=0}^{T} (\gamma\lambda)^l \delta_{t+l}, \qquad \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

$\lambda = 0.95$ controls bias-variance tradeoff. $\lambda \to 0$ approximates TD (low variance, high bias); $\lambda \to 1$ approximates Monte Carlo (high variance, low bias).

### 7.3 Value Function Loss

$$L^{\text{VF}}(\phi) = \mathbb{E}_t\!\left[\bigl(V_\phi(s_t) - V_t^{\text{target}}\bigr)^2\right]$$

### 7.4 Entropy Bonus

$$L^{\text{ENT}}(\theta) = \mathbb{E}_t\!\left[H\bigl[\pi_\theta(\cdot \mid s_t)\bigr]\right]$$

### 7.5 Total Loss

$$L(\theta, \phi) = -L^{\text{CLIP}}(\theta) + c_1 L^{\text{VF}}(\phi) - c_2 L^{\text{ENT}}(\theta)$$

with $c_1 = 0.5$, $c_2 = 0.01$.

### 7.6 Network Architecture

**Policy network** $\pi_\theta(a \mid s)$: outputs action mean $\mu$ and log-std $\log\sigma$:

```
s ∈ ℝ¹⁶  →  Dense(64, ReLU)  →  Dense(64, ReLU)  →  μ ∈ ℝ², log σ ∈ ℝ²
                                                         ↓
                                               a ~ N(μ, diag(σ²))
```

**Value network** $V_\phi(s)$:

```
s ∈ ℝ¹⁶  →  Dense(64, ReLU)  →  Dense(64, ReLU)  →  V(s) ∈ ℝ¹
```

### 7.7 Curriculum Training Stages

| Stage | Steps | Goal Radius | Time Penalty | Max Steps/Ep | Description |
|-------|-------|-------------|--------------|--------------|-------------|
| 1 | 100k | 2.0 m | −0.01 | 2000 | Learn goal-directed movement |
| 2 | 200k | 1.0 m | −0.1 | 1500 | Learn efficient paths |
| 3 | 200k | 0.5 m | −0.3 | 1000 | Precision with obstacle avoidance |

Training hyperparameters: `lr=3e-4`, `n_steps=512`, `batch_size=64`, `n_epochs=10`, `γ=0.99`, `λ=0.95`, `ε=0.2`, `ent_coef=0.02`.

---

## 8. Exploration Coverage Metric

$$C = \frac{\displaystyle\sum_{(i,j) \in \mathcal{M}} \mathbf{1}[m_{ij} = \text{free}] \cdot \delta^2}{A_{\text{building}}}$$

where $\delta = 0.05$ m/px and $A_{\text{building}} = 400$ m² (20 m × 20 m interior).

Cells are classified as **free** if occupancy probability $P(m_{ij}=1) < 0.25$ (pixel value $> 220$) and **occupied** if $P(m_{ij}=1) > 0.65$ (pixel value $< 50$). Coverage is capped at 100%.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      DISASTER ROBOT SYSTEM                        │
│                                                                    │
│   Sensors               Fusion             Planning                │
│  ┌───────┐            ┌───────┐          ┌──────────┐             │
│  │ LiDAR │──/scan────▶│       │          │  SLAM    │             │
│  │ 360°  │            │  EKF  │─/odom/──▶│ Toolbox  │──/map──▶   │
│  │ 10 Hz │            │ node  │ filtered │ (Async)  │             │
│  └───────┘            │       │          └──────────┘             │
│  ┌───────┐            │       │                                    │
│  │  IMU  │──/imu─────▶│       │          ┌──────────┐             │
│  │ 200Hz │            └───────┘          │    RL    │             │
│  └───────┘                 │             │Navigator │             │
│  ┌───────┐                 └────────────▶│ Reactive │─/cmd_vel──▶ │
│  │Camera │──/camera/──────────────────▶  │  or PPO  │             │
│  │ 30fps │   image_raw   YOLOv8          └──────────┘             │
│  └───────┘                                                         │
└──────────────────────────────────────────────────────────────────┘
```

### ROS 2 Topic Graph

| Topic | Message Type | Publisher | Subscribers |
|-------|-------------|-----------|-------------|
| `/scan` | `sensor_msgs/LaserScan` | Gazebo LiDAR plugin | SLAM Toolbox, RL Navigator |
| `/imu` | `sensor_msgs/Imu` | Gazebo IMU plugin | EKF node |
| `/odom` | `nav_msgs/Odometry` | Gazebo diff-drive plugin | EKF node |
| `/odometry/filtered` | `nav_msgs/Odometry` | EKF node (`robot_localization`) | RL Navigator, Nav2 |
| `/map` | `nav_msgs/OccupancyGrid` | SLAM Toolbox | Nav2, RViz |
| `/cmd_vel` | `geometry_msgs/Twist` | RL Navigator | Gazebo diff-drive |
| `/camera/image_raw` | `sensor_msgs/Image` | Gazebo camera plugin | YOLOv8 processor |

---

## Robot Platform

| Property | Value |
|----------|-------|
| Base | TurtleBot3 Waffle |
| Drive | Differential |
| Wheel separation | 287 mm |
| Wheel radius | 33 mm |
| Max linear velocity | 0.22 m/s |
| Max angular velocity | 1.82 rad/s |
| LiDAR | 360°, 12 m range, 10 Hz, σ = 0.01 m |
| Camera | 640 × 480 px, 30 fps, 80° FOV |
| IMU | 200 Hz, gyro σ = 2×10⁻⁴ rad/s |
| Spawn position | (0, −2, 1.0) m, yaw = π/2 |

---

## Installation

### Prerequisites

- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill
- Gazebo Classic 11
- Python 3.10+
- GPU recommended for PPO training (CPU supported)

### Step 1: Install ROS 2 Humble

```bash
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

sudo apt install software-properties-common curl -y
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu \
    $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update && sudo apt install ros-humble-desktop
```

### Step 2: Install Dependencies

```bash
sudo apt install -y \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-turtlebot3 \
    ros-humble-turtlebot3-simulations \
    ros-humble-slam-toolbox \
    ros-humble-robot-localization \
    ros-humble-nav2-bringup \
    ros-humble-nav2-map-server \
    python3-colcon-common-extensions

pip install --break-system-packages \
    stable-baselines3 gymnasium torch \
    ultralytics opencv-python-headless \
    Pillow pyyaml numpy matplotlib pandas
```

### Step 3: Build Workspace

```bash
mkdir -p ~/disaster-robot-sim/src
cd ~/disaster-robot-sim/src
git clone https://github.com/newton-adhikari/disaster-robot-sim.git .
cd ~/disaster-robot-sim

colcon build --symlink-install
source install/setup.bash
echo "export TURTLEBOT3_MODEL=waffle" >> ~/.bashrc
source ~/.bashrc
```

---

## Usage

### Launch Full Simulation

```bash
# Terminal 1 — full stack (Gazebo + EKF + SLAM + Nav2 + RViz)
export TURTLEBOT3_MODEL=waffle
ros2 launch disaster_navigation full_simulation.launch.py
```

Wait ~20 seconds for full startup sequence: Gazebo (0s) → Robot spawn (3s) → EKF (5s) → SLAM (12s) → Nav2 (17s).

### Start Autonomous Navigation

```bash
# Terminal 2 — reactive exploration
source ~/disaster-robot-sim/install/setup.bash
ros2 launch disaster_navigation rl_navigator.launch.py

# Goal-directed mode (navigate to survivor marker):
ros2 launch disaster_navigation rl_navigator.launch.py \
    mode:=navigate goal_x:=8.0 goal_y:=12.0
```

### Send Navigation Goals from RViz

Click **"2D Goal Pose"** in the RViz toolbar, then click any point on the map. The robot switches to `NAVIGATE` mode automatically.

### Save and Analyse the Map

```bash
ros2 run nav2_map_server map_saver_cli -f ~/disaster_map
python3 src/disaster_sensors/disaster_sensors/measure_coverage.py ~/disaster_map.pgm
```

---

## Running Experiments

> **Critical:** Always restart the full simulation (`Ctrl+C` → relaunch) between trials to ensure independent SLAM maps. Running a new policy on an existing map inflates coverage numbers.

### Experiment 1 — EKF Validation

```bash
# Simulation + reactive navigator must be running first
python3 src/disaster_sensors/disaster_sensors/collect_ekf_data.py --duration 300
# Outputs: ~/ekf_comparison.csv, ~/ekf_trajectory.png
# Prints Table I (RMSE comparison) to stdout
```

### Experiment 2 — Reactive Policy (3 trials)

```bash
# Fresh sim restart between each trial
timeout 300 ros2 launch disaster_navigation rl_navigator.launch.py
ros2 run nav2_map_server map_saver_cli -f ~/map_reactive_trial1
python3 src/disaster_sensors/disaster_sensors/measure_coverage.py ~/map_reactive_trial1.pgm
# Repeat for trial2, trial3
```

### Experiment 3 — Random Walk Baseline (3 trials)

```bash
# Patch navigator
python3 src/disaster_sensors/disaster_sensors/random_walk_patch.py patch
colcon build --symlink-install && source install/setup.bash

# Run 3 trials with fresh sim restart between each
timeout 300 ros2 launch disaster_navigation rl_navigator.launch.py
ros2 run nav2_map_server map_saver_cli -f ~/map_random_trial1
python3 src/disaster_sensors/disaster_sensors/measure_coverage.py ~/map_random_trial1.pgm

# Restore reactive when done
python3 src/disaster_sensors/disaster_sensors/random_walk_patch.py restore
colcon build --symlink-install && source install/setup.bash
```

### Experiment 4 — PPO Training (see limitation below)

```bash
# Terminal 1: launch without RViz to save CPU
ros2 launch disaster_navigation full_simulation.launch.py use_rviz:=false

# Terminal 2: curriculum training
python3 disaster_rl_trainer/scripts/train_curriculum.py

# Monitor with TensorBoard
tensorboard --logdir=disaster_rl_trainer/models/tensorboard_curriculum/
```

---

## PPO Training — Known Limitation

PPO curriculum training was implemented and executed. Training progressed correctly for ~16,000 steps, with mean episode reward rising from −37 to +32. At step ~16,896, training collapsed to degenerate single-step episodes (`ep_rew_mean` deteriorating to −1,300, `value_loss` reaching 10⁶).

**Root cause:** Gazebo Classic's `SetEntityState` service zeros commanded velocity at episode reset but does not flush the internal broadphase collision cache. Residual contact forces from the previous episode persist, causing an immediate −100 collision penalty on step 1 of each new episode. This creates a degenerate feedback loop that destabilises the value function.

**Why the standard fix doesn't work:** The `/reset_world` service resolves the physics state correctly but requires 15–20 seconds per episode, making 100k-step training infeasible (~200+ hours wall time).

**Mitigation path for future work:** Migrate to Gazebo Ignition (Fortress/Garden), which exposes a proper `ResetWorld` gRPC API with sub-second physics reset. Alternatively, implement a headless Isaac Gym or MuJoCo port of the environment.

Until resolved, the reactive policy serves as the primary navigation contribution. The PPO negative result is reported in full as it provides actionable guidance for future researchers.

---

## Project Structure

```
disaster-robot-sim/
├── disaster_rl_trainer/             # Standalone PPO training (no ROS needed for training loop)
│   ├── disaster_rl_env/
│   │   └── disaster_nav_env.py      # Gymnasium env (16-dim obs, 2-dim action, curriculum reward)
│   ├── models/
│   │   ├── checkpoints/             # Saved model checkpoints every 20k steps
│   │   └── tensorboard_curriculum/  # TensorBoard training logs
│   └── scripts/
│       ├── train_ppo.py             # Flat PPO training (500k steps)
│       └── train_curriculum.py      # 3-stage curriculum training (recommended)
│
├── src/
│   ├── disaster_robot/              # Robot URDF and hardware description
│   │   ├── urdf/
│   │   │   └── disaster_robot.urdf.xacro
│   │   └── launch/
│   │       └── spawn_robot.launch.py
│   │
│   ├── disaster_world/              # Gazebo environment
│   │   └── worlds/
│   │       └── disaster_collapsed_building.world
│   │           # 20m × 20m building:
│   │           #   4 collapsed walls, 5 rubble piles,
│   │           #   3 broken pillars, 8 debris pieces,
│   │           #   survivor marker (goal) at (8, 12)
│   │
│   ├── disaster_navigation/         # Navigation stack config
│   │   ├── config/
│   │   │   ├── ekf_params.yaml          # EKF sensor fusion config (Q matrix, sensor masks)
│   │   │   ├── nav2_params.yaml         # Nav2 DWB planner config
│   │   │   └── slam_toolbox_params.yaml # Async SLAM parameters
│   │   ├── launch/
│   │   │   ├── full_simulation.launch.py    # Main launch (timed sequence)
│   │   │   └── rl_navigator.launch.py       # Navigator launch
│   │   └── rviz/
│   │       └── disaster_full.rviz
│   │
│   └── disaster_sensors/            # Python sensor processing nodes
│       └── disaster_sensors/
│           ├── rl_navigator.py          # Reactive FSM navigation agent
│           ├── camera_processor.py      # YOLOv8 survivor detection
│           ├── ekf_monitor.py           # EKF diagnostics and CSV logging
│           ├── lidar_processor.py       # LiDAR gap detection utility
│           ├── collect_ekf_data.py      # EKF vs odometry comparison (Experiment 1)
│           ├── random_walk_patch.py     # Patches rl_navigator for random walk baseline
│           ├── measure_coverage.py      # Map coverage analysis from .pgm files
│           └── training/
│               └── train_disaster_yolov8.py  # YOLOv8 fine-tuning on custom dataset
│
└── scripts/
    ├── record_demo.py               # Record EKF trajectory data to bag
    └── plot_ekf_results.py          # Generate trajectory comparison figures
```

---

## Common Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `libexec directory does not exist` | Missing `setup.cfg` | Ensure `setup.cfg` has `[install] install_scripts=$base/lib/disaster_sensors` |
| `timestamp earlier than transform cache` | SLAM starts before EKF TF stabilises | Normal during startup — clears after ~30s. SLAM is delayed to t=12s in launch file |
| Robot doesn't move | Stale `install/` from previous build | `rm -rf build/ install/ log/` then `colcon build --symlink-install` |
| RViz entirely purple | Fixed Frame = `map` before SLAM builds TF | Change Fixed Frame to `odom` in RViz → Global Options |
| LiDAR reads 0.12–0.15 m everywhere | Self-hits from chassis geometry | LiDAR `min_range` set to 0.25 m in URDF — already handled |
| Coverage > 100% | Sim not restarted between trials | Always kill and relaunch `full_simulation.launch.py` between trials |
| PPO collapses to ep_len=1 | Gazebo physics reset bug | See [PPO Training — Known Limitation](#ppo-training--known-limitation) |
| `/odometry/filtered` receives 0 messages | QoS mismatch | Use default QoS (depth=10) not BEST_EFFORT when subscribing |

---

## Citation

```bibtex
@misc{adhikari2025disaster,
  author    = {Newton Adhikari},
  title     = {Adaptive Autonomous Navigation for Disaster Response Robots
               Using Multi-Modal Sensor Fusion and Reinforcement Learning},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/newton-adhikari/disaster-robot-simulator}
}
```

---

## Related Work

- [RL Goal Navigation TurtleBot3](https://github.com/newton-adhikari/rl_goal_nav_tb3) — PPO-based goal navigation in structured environments (92% success rate), precursor to this project
- [SLAM Toolbox](https://github.com/SteveMacenski/slam_toolbox) — Async pose-graph SLAM used for real-time mapping
- [robot_localization](https://github.com/cra-ros-pkg/robot_localization) — EKF/UKF node for sensor fusion
- [Nav2](https://navigation.ros.org/) — Navigation stack for goal-directed waypoint following
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) — PPO implementation used for training

## Acknowledgments

- ROS 2 and Gazebo Classic development teams
- TurtleBot3 Robotis team for the open robot platform
- Steve Macenski (SLAM Toolbox) for async pose-graph SLAM
- Tom Moore (robot_localization) for the EKF implementation