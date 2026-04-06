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


# ── Helper functions ──────────────────────────────────────────────────────────
log() {
    local msg="[$(date +%H:%M:%S)] $*"
    echo "$msg"
    echo "$msg" >> "$LOG"
}

sep() {
    local line="────────────────────────────────────────────"
    echo "$line"
    echo "$line" >> "$LOG"
}

wait_for_port_free() {
    local waited=0
    while true; do
        if ! (ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null) \
                | grep -q ":${GZ_PORT} " 2>/dev/null; then
            break
        fi
        if [ "$waited" -ge "$PORT_WAIT_MAX" ]; then
            log "WARNING: port $GZ_PORT still in use after ${PORT_WAIT_MAX}s"
            return
        fi
        sleep 2
        waited=$((waited + 2))
    done
}

kill_sim() {
    log "Killing simulation..."
    if [ -n "$SIM_PID" ]; then
        kill -- -"$SIM_PID" 2>/dev/null; true
        kill "$SIM_PID"     2>/dev/null; true
    fi
    sleep 2
    pkill -f gzserver          2>/dev/null; true
    pkill -f gzclient          2>/dev/null; true
    pkill -f robot_state_pub   2>/dev/null; true
    pkill -f slam_toolbox      2>/dev/null; true
    pkill -f controller_server 2>/dev/null; true
    pkill -f planner_server    2>/dev/null; true
    pkill -f behavior_server   2>/dev/null; true
    pkill -f bt_navigator      2>/dev/null; true
    pkill -f lifecycle_manager 2>/dev/null; true
    pkill -f ekf_node          2>/dev/null; true
    pkill -f ekf_monitor       2>/dev/null; true
    pkill -f auto_map_saver    2>/dev/null; true
    pkill -f frontier_explorer 2>/dev/null; true
    pkill -f potential_field   2>/dev/null; true
    pkill -f rl_navigator      2>/dev/null; true
    pkill -f spawn_entity      2>/dev/null; true
    sleep 2
    pkill -9 -f gzserver          2>/dev/null; true
    pkill -9 -f gzclient          2>/dev/null; true
    pkill -9 -f slam_toolbox      2>/dev/null; true
    pkill -9 -f controller_server 2>/dev/null; true
    pkill -9 -f planner_server    2>/dev/null; true
    pkill -9 -f bt_navigator      2>/dev/null; true
    pkill -9 -f lifecycle_manager 2>/dev/null; true
    pkill -9 -f ekf_node          2>/dev/null; true
    sleep "$KILL_SETTLE"
    wait_for_port_free
    SIM_PID=""
    log "Simulation stopped."
}

# ── Run one trial ─────────────────────────────────────────────────────────────
# Everything runs inside ros2 launch — no separate ros2 run calls.
SIM_PID=""
LAST_COVERAGE="N/A"