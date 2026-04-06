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


run_trial() {
    local nav_policy="$1"   # frontier_explorer | potential_field_navigator | rl_navigator | rl_navigator_model
    local label="$2"
    local policy_name="$3"
    local trial_num="$4"

    sep
    log "Starting trial: $nav_policy → $label"
    sep

    local ekf_csv="$RUN_DIR/${label}_ekf.csv"
    local coll_csv="$RUN_DIR/${label}_collisions.csv"

    log "Launching simulation + policy (headless)..."
    ros2 launch disaster_navigation full_simulation.launch.py \
        use_rviz:=false gui:=false \
        map_output_dir:="$RUN_DIR" \
        map_output_prefix:="$label" \
        nav_policy:="$nav_policy" \
        rl_model_path:="$RL_MODEL_PATH" \
        launch_ekf_monitor:=true \
        ekf_csv:="$ekf_csv" \
        collision_csv:="$coll_csv" \
        >> "$LOG" 2>&1 &
    SIM_PID=$!
    log "Simulation PID=$SIM_PID"

    # Wait: warmup + trial duration
    local total_wait=$((SIM_WARMUP + TRIAL_DURATION))
    log "Waiting ${SIM_WARMUP}s warmup + ${TRIAL_DURATION}s trial = ${total_wait}s..."
    sleep "$total_wait"
    log "Trial time complete."

    # Check map
    local map_path="$RUN_DIR/$label"
    if [ -f "${map_path}.pgm" ]; then
        log "Map file confirmed: ${map_path}.pgm"
    else
        log "WARNING: map not found at ${map_path}.pgm"
    fi

    # Compute coverage
    if [ -f "${map_path}.pgm" ]; then
        LAST_COVERAGE=$(python3 -c "
from disaster_sensors.benchmark_metrics import BenchmarkMetrics
bm = BenchmarkMetrics(verbose=False)
result = bm.compute_coverage('${map_path}.pgm')
print(f\"{result['coverage_pct']:.1f}\")
" 2>/dev/null || echo "N/A")
        log "Coverage: ${LAST_COVERAGE}%  (${label})"
    else
        LAST_COVERAGE="N/A"
    fi

    # Generate metrics JSON
    local json_out="$RUN_DIR/${label}_metrics.json"
    if [ -f "$ekf_csv" ] && [ -f "$coll_csv" ] && [ -f "${map_path}.pgm" ]; then
        python3 -m disaster_sensors.benchmark_metrics \
            --map "${map_path}.pgm" \
            --ekf-log "$ekf_csv" \
            --collision-log "$coll_csv" \
            --duration "$TRIAL_DURATION" \
            --policy "$policy_name" \
            --trial "$trial_num" \
            --output "$json_out" \
            >> "$LOG" 2>&1
        log "Metrics JSON: $json_out"
    else
        python3 -c "
import json
data = {
    'policy': '${policy_name}',
    'trial': ${trial_num},
    'duration_s': ${TRIAL_DURATION},
    'coverage': {'coverage_pct': ${LAST_COVERAGE:-0}},
}
with open('${json_out}', 'w') as f:
    json.dump(data, f, indent=2)
" 2>/dev/null
        log "Coverage-only JSON: $json_out"
    fi

    kill_sim
}

# ── Storage ───────────────────────────────────────────────────────────────────
declare -a FRONTIER_COV POTFIELD_COV REACTIVE_COV RL_COV

# ── Pre-run cleanup ──────────────────────────────────────────────────────────
log "Pre-run cleanup..."
pkill -9 -f gzserver      2>/dev/null; true
pkill -9 -f gzclient      2>/dev/null; true
pkill -9 -f slam_toolbox  2>/dev/null; true
sleep 3
wait_for_port_free
log "Pre-run cleanup done."
echo ""

# =============================================================================
sep
log "PHASE 1: Frontier Explorer — ${N_TRIALS} trials"
sep

for i in $(seq 1 $N_TRIALS); do
    log "Frontier trial $i / $N_TRIALS"
    run_trial "frontier_explorer" "frontier_trial${i}" "frontier" "$i"
    FRONTIER_COV+=("$LAST_COVERAGE")
    log "Frontier trial $i result: ${LAST_COVERAGE}%"
done

# =============================================================================
sep
log "PHASE 2: Potential Field — ${N_TRIALS} trials"
sep

for i in $(seq 1 $N_TRIALS); do
    log "Potential field trial $i / $N_TRIALS"
    run_trial "potential_field_navigator" "potfield_trial${i}" "potential_field" "$i"
    POTFIELD_COV+=("$LAST_COVERAGE")
    log "Potential field trial $i result: ${LAST_COVERAGE}%"
done

# =============================================================================
sep
log "PHASE 3: Reactive FSM — ${N_TRIALS} trials"
sep

for i in $(seq 1 $N_TRIALS); do
    log "Reactive FSM trial $i / $N_TRIALS"
    run_trial "rl_navigator" "reactive_trial${i}" "reactive_fsm" "$i"
    REACTIVE_COV+=("$LAST_COVERAGE")
    log "Reactive FSM trial $i result: ${LAST_COVERAGE}%"
done

# =============================================================================
if [ -f "$RL_MODEL_PATH" ]; then
    sep
    log "PHASE 4: RL Navigator — ${N_TRIALS} trials"
    sep

    for i in $(seq 1 $N_TRIALS); do
        log "RL trial $i / $N_TRIALS"
        run_trial "rl_navigator_model" "rl_trial${i}" "rl_navigator" "$i"
        RL_COV+=("$LAST_COVERAGE")
        log "RL trial $i result: ${LAST_COVERAGE}%"
    done
else
    log "PHASE 4: SKIPPED — no RL model at $RL_MODEL_PATH"
fi

# =============================================================================
sep
log "ALL TRIALS COMPLETE"
sep


# ── Generate aggregate report ─────────────────────────────────────────────────
python3 - "$RUN_DIR" \
    "${FRONTIER_COV[@]:-}" "---" \
    "${POTFIELD_COV[@]:-}" "---" \
    "${REACTIVE_COV[@]:-}" "---" \
    "${RL_COV[@]:-}" <<'PYEOF'
import sys, os, json
import numpy as np
from scipy import stats as sp_stats

run_dir = sys.argv[1]
args = sys.argv[2:]

groups, current = [], []
for a in args:
    if a == "---":
        groups.append(current); current = []
    elif a and a != "N/A":
        current.append(float(a))
groups.append(current)

names = ["Frontier Explorer", "Potential Field", "Reactive FSM", "RL Navigator"]

def row(name, vals):
    if not vals:
        return f"  {name:<22} NO DATA"
    a = np.array(vals)
    m, s = a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0
    ci = 1.96 * s / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return (f"  {name:<22} n={len(a):2d}  mean={m:5.1f}%  "
            f"std={s:4.1f}%  95%CI=[{m-ci:.1f}, {m+ci:.1f}]")

print()
print("╔══════════════════════════════════════════════════════════════════════╗")
print("║               DisasterSim Results Summary                      ║")
print("╠══════════════════════════════════════════════════════════════════════╣")
for name, vals in zip(names, groups):
    print(row(name, vals))

print("╠══════════════════════════════════════════════════════════════════════╣")
print("  Pairwise Mann-Whitney U tests (coverage):")
for i in range(len(groups)):
    for j in range(i+1, len(groups)):
        if len(groups[i]) >= 3 and len(groups[j]) >= 3:
            u_stat, p_val = sp_stats.mannwhitneyu(
                groups[i], groups[j], alternative='two-sided')
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"    {names[i]:20s} vs {names[j]:20s}: U={u_stat:.0f} p={p_val:.4f} {sig}")

print("╚══════════════════════════════════════════════════════════════════════╝")

aggregate = {}
for name, vals in zip(names, groups):
    if vals:
        a = np.array(vals)
        aggregate[name] = {
            "n": len(vals), "mean": round(float(a.mean()), 2),
            "std": round(float(a.std(ddof=1)), 2) if len(a) > 1 else 0.0,
            "min": round(float(a.min()), 2), "max": round(float(a.max()), 2),
            "values": [round(v, 2) for v in vals],
        }

agg_path = os.path.join(run_dir, "aggregate_results.json")
with open(agg_path, 'w') as f:
    json.dump(aggregate, f, indent=2)
print(f"\n  Aggregate JSON: {agg_path}\n")
PYEOF

log "Done. All results saved to $RUN_DIR"
log "Log file: $LOG"
