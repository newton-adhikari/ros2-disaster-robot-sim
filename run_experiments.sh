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


# ── Source ROS and workspace ──────────────────────────────────────────────────
echo "[setup] Sourcing ROS 2 Humble..."
source /opt/ros/humble/setup.bash
echo "[setup] Sourcing workspace..."
source "$REPO/install/setup.bash"
export TURTLEBOT3_MODEL=waffle
export FASTRTPS_DEFAULT_PROFILES_FILE="$REPO/config/fastdds_localhost.xml"
export ROS_LOCALHOST_ONLY=1
echo "[setup] Done. TURTLEBOT3_MODEL=waffle"
echo ""

# ── Create results directory ──────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$RESULTS/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/experiment.log"
echo "[setup] Results → $RUN_DIR"
echo "[setup] Log     → $LOG"
echo ""
