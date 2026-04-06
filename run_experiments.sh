#!/usr/bin/env bash
#
# All policy nodes, EKF monitor, and map saver run INSIDE ros2 launch
# on WSL2.


echo "========================================"
echo "  DisasterSim benchmark Experiment Runner"
echo "========================================"
echo ""

# ── Config ────────────────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$HOME/disaster_results"
TRIAL_DURATION=300
SIM_WARMUP=100
N_TRIALS=10
KILL_SETTLE=8
GZ_PORT=11345
PORT_WAIT_MAX=30

RL_MODEL_PATH="$REPO/disaster_rl_trainer/scripts/models/disaster_ppo_v2_final.zip"
