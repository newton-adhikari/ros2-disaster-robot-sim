#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

import numpy as np
from scipy.ndimage import label as scipy_label
from scipy.ndimage import binary_dilation
import math, time, os, threading


# the basic idead of this class is
# Subscribe to the occupancy grid
# Detect free cells adjacent to unknown cells
# Cluster those frontiers and select the nearest centroid
# Navigate to them using Nav2 SimpleCommander


class FrontierExplorer(Node):

    FREE_THRESH         = 25
    UNKNOWN_VAL         = -1
    MIN_FRONTIER_CELLS  = 8       # was 5 — filter tiny noise clusters
    MIN_FRONTIER_DIST_M = 0.8     # lowered from 1.2 — let Nav2 handle close goals
    MAX_FRONTIER_DIST_M = 15.0    # raised from 12
    GOAL_RADIUS_M       = 0.7
    NAV_TIMEOUT_S       = 45.0    # raised from 30 — give Nav2 more time .....
    STUCK_TIME_S        = 12.0
    STUCK_VEL_THR       = 0.02
    MAX_NAV2_FAILURES   = 3
    UPDATE_HZ           = 2.0
    NAV2_WAIT_TIMEOUT_S = 90.0

    COOLDOWN_STEPS      = 10      # ignore a reached frontier for N ticks
    REVISIT_RADIUS_M    = 1.5     # skip goals within this of any visited goal
    SCORE_SIZE_WEIGHT   = 0.6     # weight for frontier size in scoring
    SCORE_DIST_WEIGHT   = 0.4     # weight for distance in scoring

    DRIVE_SPEED         = 0.18    # raised from 0.15
    TURN_SPEED          = 0.9
    FRONT_ARC_DEG       = 30
    FRONT_OBSTACLE_M    = 0.45

    # of no-progress
    NO_PROGRESS_TIMEOUT_S = 45.0  # if no new coverage for this long, reset filters
    MAX_BLACKLIST_SIZE    = 15    # cap blacklist to prevent over-filtering
    WANDER_DURATION_S     = 10.0  # wander for this long when no frontiers found

    def __init__(self):
        super().__init__('frontier_explorer')

        os.makedirs(os.path.expanduser('~/disaster-lab/disaster_results'), exist_ok=True)

        self._nav2_ready      = False
        self._nav2_wait_done  = False

        self._map             = None
        self._robot_x         = 0.0
        self._robot_y         = 0.0
        self._robot_yaw       = 0.0
        self._scan_ranges     = []
        self._scan_angle_min  = -math.pi
        self._scan_angle_inc  = math.radians(1.0)
        self._navigating      = False
        self._goal_handle     = None
        self._goal_sent_time  = 0.0
        self._last_motion_t   = time.time()
        self._current_goal_xy = None
        self._blacklist       = set()
        self._nav2_fail_count = 0
        self._fallback_goal   = None

        # anti-cycling state
        self._visited_goals   = []        # list of (x, y) reached goals
        self._cooldown_set    = {}        # key -> remaining cooldown ticks
        self._goals_sent      = 0
        self._goals_reached   = 0

        # check no-progress 
        self._last_frontier_count = 0
        self._last_progress_t     = time.time()
        self._wander_until        = 0.0   # timestamp: wander until this time

        # QoS
        latch_qos = QoSProfile(depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        be_qos = QoSProfile(depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT)

        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.create_subscription(OccupancyGrid, '/map',               self._map_cb,  latch_qos)
        self.create_subscription(Odometry,      '/odom', self._odom_cb, be_qos)
        self.create_subscription(LaserScan,     '/scan',              self._scan_cb, be_qos)
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(1.0 / self.UPDATE_HZ, self._explore_step)

        threading.Thread(target=self._wait_for_nav2, daemon=True).start()

        self.get_logger().info(
            f'FrontierExplorer  ready. '
            f'Waiting up to {self.NAV2_WAIT_TIMEOUT_S}s for Nav2...')

    # Nav2 background wait

    def _wait_for_nav2(self):
        ready = self._nav_client.wait_for_server(
            timeout_sec=self.NAV2_WAIT_TIMEOUT_S)
        if ready:
            self._nav2_ready = True
            self.get_logger().info('Nav2 action server READY — switching to frontier mode.')
        else:
            self.get_logger().warn(
                f'Nav2 did not become ready within {self.NAV2_WAIT_TIMEOUT_S}s. '
                f'Will run in direct-drive mode only.')
        self._nav2_wait_done = True

    def _map_cb(self, msg):
        self._map = msg

    def _odom_cb(self, msg):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2*(q.w*q.z + q.x*q.y)
        cosy = 1 - 2*(q.y*q.y + q.z*q.z)
        self._robot_yaw = math.atan2(siny, cosy)
        vx = msg.twist.twist.linear.x
        wz = msg.twist.twist.angular.z
        if abs(vx) > self.STUCK_VEL_THR or abs(wz) > self.STUCK_VEL_THR:
            self._last_motion_t = time.time()

    def _scan_cb(self, msg):
        self._scan_ranges    = list(msg.ranges)
        self._scan_angle_min = msg.angle_min
        self._scan_angle_inc = msg.angle_increment

    # ── LiDAR helper functions

    def _front_obstacle(self):
        if not self._scan_ranges:
            return False
        arc = math.radians(self.FRONT_ARC_DEG)
        for i, r in enumerate(self._scan_ranges):
            if math.isnan(r) or math.isinf(r):
                continue
            angle = self._scan_angle_min + i * self._scan_angle_inc
            if abs(angle) <= arc and r < self.FRONT_OBSTACLE_M:
                return True
        return False

    # ── main frontier detection

    def _detect_frontiers(self):
        if self._map is None:
            return []
        w   = self._map.info.width
        h   = self._map.info.height
        res = self._map.info.resolution
        ox  = self._map.info.origin.position.x
        oy  = self._map.info.origin.position.y
        grid = np.array(self._map.data, dtype=np.int16).reshape(h, w)

        free    = (grid >= 0) & (grid < self.FREE_THRESH)
        unknown = (grid == self.UNKNOWN_VAL)
        struct  = np.array([[0,1,0],[1,1,1],[0,1,0]])
        mask    = free & binary_dilation(unknown, structure=struct)

        labeled, n = scipy_label(mask)
        centroids = []
        for cid in range(1, n+1):
            cells = np.argwhere(labeled == cid)
            if len(cells) < self.MIN_FRONTIER_CELLS:
                continue
            cy = cells[:,0].mean() * res + oy
            cx = cells[:,1].mean() * res + ox
            d  = math.hypot(cx - self._robot_x, cy - self._robot_y)
            if d < self.MIN_FRONTIER_DIST_M or d > self.MAX_FRONTIER_DIST_M:
                continue
            key = (round(cx, 1), round(cy, 1))
            if key in self._blacklist:
                continue
            # skip if in cooldown
            if key in self._cooldown_set:
                continue
            # skip if too close to a previously visited goal
            if self._near_visited(cx, cy):
                continue
            centroids.append((cx, cy, len(cells), d))
        return centroids

    def _near_visited(self, x, y):
        # Check if (x,y) is within REVISIT_RADIUS_M of any visited goal.
        for vx, vy in self._visited_goals:
            if math.hypot(x - vx, y - vy) < self.REVISIT_RADIUS_M:
                return True
        return False

    def _score_frontier(self, size, dist):
        # Score a frontier: prefer large frontiers that aren't too far

        # Normalise size to [0,1] range (cap at 100 cells)
        size_norm = min(size / 100.0, 1.0)
        # Normalise distance inversely
        dist_norm = 1.0 / (1.0 + dist)
        return (self.SCORE_SIZE_WEIGHT * size_norm +
                self.SCORE_DIST_WEIGHT * dist_norm)

    # Nav2 goal sending

    def _send_nav2_goal(self, x, y):
        goal = NavigateToPose.Goal()
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        yaw = math.atan2(y - self._robot_y, x - self._robot_x)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)
        goal.pose.pose.orientation.z = math.sin(yaw / 2)

        future = self._nav_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_cb)
        self._goal_sent_time  = time.time()
        self._navigating      = True
        self._current_goal_xy = (x, y)
        self._goals_sent     += 1
        self.get_logger().info(f'Nav2 goal #{self._goals_sent} → ({x:.2f}, {y:.2f})')

    def _goal_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Goal rejected.')
            self._navigating = False
            self._on_nav2_failure()
            return
        self._goal_handle = handle
        handle.get_result_async().add_done_callback(self._goal_result_cb)

    def _goal_result_cb(self, future):
        status = future.result().status
        self._navigating  = False
        self._goal_handle = None
        if status == GoalStatus.STATUS_SUCCEEDED:
            self._goals_reached += 1
            self.get_logger().info(
                f'Frontier reached ({self._goals_reached}/{self._goals_sent}).')
            self._nav2_fail_count = 0
            self._fallback_goal   = None

            # record visited goal and add cooldown
            if self._current_goal_xy:
                self._visited_goals.append(self._current_goal_xy)
                key = (round(self._current_goal_xy[0], 1),
                       round(self._current_goal_xy[1], 1))
                self._cooldown_set[key] = self.COOLDOWN_STEPS
        else:
            self.get_logger().warn(f'Goal status={status}')
            self._on_nav2_failure()

    def _on_nav2_failure(self):
        self._nav2_fail_count += 1
        if self._current_goal_xy and \
                self._nav2_fail_count >= self.MAX_NAV2_FAILURES:
            key = (round(self._current_goal_xy[0], 1),
                   round(self._current_goal_xy[1], 1))
            self._blacklist.add(key)
            self.get_logger().warn(
                f'Blacklisted {key}  (blacklist={len(self._blacklist)})')
            self._nav2_fail_count = 0
            self._current_goal_xy = None
            self._fallback_goal   = None
        elif self._current_goal_xy:
            self._fallback_goal = self._current_goal_xy

    def _cancel_current_goal(self):
        if self._goal_handle:
            self._goal_handle.cancel_goal_async()
            self._goal_handle = None
        self._navigating = False

    # Direct-drive helper functions

    def _direct_drive_toward(self, gx, gy):
        dist = math.hypot(gx - self._robot_x, gy - self._robot_y)
        if dist < self.GOAL_RADIUS_M:
            self._stop()
            return True

        cmd = Twist()
        if self._front_obstacle():
            cmd.angular.z = self.TURN_SPEED
        else:
            angle_err = math.atan2(gy - self._robot_y, gx - self._robot_x) \
                        - self._robot_yaw
            while angle_err >  math.pi: angle_err -= 2*math.pi
            while angle_err < -math.pi: angle_err += 2*math.pi
            cmd.angular.z = min(self.TURN_SPEED,
                                max(-self.TURN_SPEED, 1.5 * angle_err))
            cmd.linear.x  = self.DRIVE_SPEED * max(0.0,
                                1.0 - abs(angle_err) / math.pi)
        self._cmd_pub.publish(cmd)
        return False

    def _stop(self):
        self._cmd_pub.publish(Twist())

    def _wander(self):
        # Active wandering with obstacle avoidance and random direction changes
        cmd = Twist()
        if self._front_obstacle():
            # Turn away from obstacle — pick direction with more space
            if self._scan_ranges:
                n = len(self._scan_ranges)
                left_min = min(self._scan_ranges[n//6 : n//3] or [10.0])
                right_min = min(self._scan_ranges[2*n//3 : 5*n//6] or [10.0])
                cmd.angular.z = self.TURN_SPEED if left_min >= right_min else -self.TURN_SPEED
            else:
                cmd.angular.z = self.TURN_SPEED
        else:
            cmd.linear.x = self.DRIVE_SPEED
            # Gentle random drift to avoid straight-line loops
            cmd.angular.z = np.random.uniform(-0.3, 0.3)
        self._cmd_pub.publish(cmd)

    # Main control loop

    def _explore_step(self):

        # Tick down cooldowns
        expired = [k for k, v in self._cooldown_set.items() if v <= 1]
        for k in expired:
            del self._cooldown_set[k]
        for k in self._cooldown_set:
            self._cooldown_set[k] -= 1

        #  no-progress watchdog — if stuck too long, reset filters
        now = time.time()
        if self._map is not None:
            grid = np.array(self._map.data, dtype=np.int16)
            free_count = int(np.sum((grid >= 0) & (grid < self.FREE_THRESH)))
            if free_count > self._last_frontier_count + 20:
                self._last_frontier_count = free_count
                self._last_progress_t = now
            elif now - self._last_progress_t > self.NO_PROGRESS_TIMEOUT_S:
                self.get_logger().warn(
                    f'No progress for {self.NO_PROGRESS_TIMEOUT_S}s — '
                    f'clearing blacklist ({len(self._blacklist)}) and '
                    f'visited ({len(self._visited_goals)})')
                self._blacklist.clear()
                self._visited_goals.clear()
                self._cooldown_set.clear()
                self._nav2_fail_count = 0
                self._last_progress_t = now

        # Phase 1: Nav2 not yet ready → wander
        if not self._nav2_ready:
            if not self._nav2_wait_done:
                self._wander()
            return

        # if in timed wander mode, keep wandering
        if now < self._wander_until:
            self._wander()
            return

        # Phase 2: Nav2 ready

        # Handle direct-drive fallback goal
        if self._fallback_goal is not None and not self._navigating:
            gx, gy = self._fallback_goal
            arrived = self._direct_drive_toward(gx, gy)
            if arrived:
                self._fallback_goal   = None
                self._nav2_fail_count = 0
            return

        if self._navigating:
            elapsed   = time.time() - self._goal_sent_time
            stuck_for = time.time() - self._last_motion_t
            if elapsed > self.NAV_TIMEOUT_S:
                self.get_logger().warn('Nav2 timeout — replanning.')
                self._cancel_current_goal()
                self._on_nav2_failure()
            elif stuck_for > self.STUCK_TIME_S:
                self.get_logger().warn('Stuck — replanning.')
                self._cancel_current_goal()
                self._on_nav2_failure()
            return

        #  cap blacklist size
        if len(self._blacklist) > self.MAX_BLACKLIST_SIZE:
            self.get_logger().info('Blacklist full — clearing oldest entries.')
            self._blacklist.clear()

        # Select next frontier (: score-based, not nearest-first)
        centroids = self._detect_frontiers()
        if not centroids:
            #  wander actively instead of spinning in place
            self.get_logger().info(
                f'No valid frontiers — wandering for {self.WANDER_DURATION_S}s '
                f'(blacklist={len(self._blacklist)}, visited={len(self._visited_goals)})')
            self._wander_until = now + self.WANDER_DURATION_S
            self._wander()
            return

        # Score each frontier and pick the best
        scored = [(self._score_frontier(size, dist), x, y, size, dist)
                  for x, y, size, dist in centroids]
        scored.sort(key=lambda c: c[0], reverse=True)
        _, x, y, size, dist = scored[0]

        self.get_logger().info(
            f'Frontiers: {len(centroids)} | '
            f'→ ({x:.1f},{y:.1f}) size={size} dist={dist:.1f}m '
            f'blacklist={len(self._blacklist)} visited={len(self._visited_goals)}')
        self._send_nav2_goal(x, y)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()