#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist

from scipy.ndimage import label as scipy_label
import numpy as np
import math
import time


# explore unknown environments safely
# to make the robot move autonomously, even without a full path planner or RL agent
# it uses reactive navigation algorithm

# Attractive force → toward the centroid of unexplored space (encourages exploration).
# Repulsive force → away from obstacles detected by LiDAR (avoids collisions).


class PotentialFieldNavigator(Node):

    K_ATT       = 2.0     # raised from 1.0 — stronger pull toward unknown
    K_REP       = 0.4     # lowered from 0.5 — less obstacle repulsion
    D_REP       = 1.2     # lowered from 1.5 — tighter repulsive radius
    D_SAFE      = 0.30    # lowered from 0.35 — allow closer approach
    VEL_MAX     = 0.20    # raised from 0.18
    OMEGA_MAX   = 1.2
    STALL_VEL   = 0.03    # raised from 0.02 — detect stalls earlier
    STALL_TIME  = 4.0     # lowered from 5.0 — react faster
    PERTURB_DUR = 4.0     # raised from 2.5 — longer escape manoeuvre
    UPDATE_HZ   = 10.0

    # Wall-following mode
    WALL_FOLLOW_TRIGGER = 3       # consecutive stalls before wall-follow
    WALL_FOLLOW_DUR     = 15.0    # seconds of wall-following
    WALL_FOLLOW_DIST    = 0.6     # desired distance from wall
    WALL_FOLLOW_SPEED   = 0.15

    # Unknown cluster detection
    MIN_CLUSTER_CELLS   = 10
    FREE_THRESH         = 25

    def __init__(self):
        super().__init__('potential_field_navigator')

        self._scan = None
        self._map = None
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_yaw = 0.0
        self._last_nonzero_vel = time.time()

        # Perturbation state
        self._perturbing = False
        self._perturb_end = 0.0
        self._perturb_dir = 1.0
        self._consecutive_stalls = 0

        # Wall-following state
        self._wall_following = False
        self._wall_follow_end = 0.0
        self._wall_follow_side = 1.0  # 1.0 = follow left wall, -1.0 = right

        # Cached nearest unknown target
        self._attract_target = None
        self._attract_update_time = 0.0

        # QoS
        latch_qos = QoSProfile(depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        best_effort_qos = QoSProfile(depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT)

        self.create_subscription(LaserScan,     '/scan',              self._scan_cb,  best_effort_qos)
        self.create_subscription(OccupancyGrid, '/map',               self._map_cb,   latch_qos)
        self.create_subscription(Odometry,      '/odom', self._odom_cb,  best_effort_qos)

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(1.0 / self.UPDATE_HZ, self._control_step)

        self.get_logger().info('PotentialFieldNavigator v2 ready.')

    # Callbacks

    def _scan_cb(self, msg):
        self._scan = msg

    def _map_cb(self, msg):
        self._map = msg

    def _odom_cb(self, msg):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0*(q.w*q.z + q.x*q.y)
        cosy = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
        self._robot_yaw = math.atan2(siny, cosy)
        vx = msg.twist.twist.linear.x
        wz = msg.twist.twist.angular.z
        if abs(vx) > self.STALL_VEL or abs(wz) > self.STALL_VEL:
            self._last_nonzero_vel = time.time()

    # Nearest unknown cluster

    def _find_nearest_unknown_target(self):
        # Find centroid of the nearest cluster of unknown cells.
        if self._map is None:
            return None

        now = time.time()
        # Cache for 2 seconds to avoid recomputing every tick
        if (self._attract_target is not None and
                now - self._attract_update_time < 2.0):
            return self._attract_target

        w   = self._map.info.width
        h   = self._map.info.height
        res = self._map.info.resolution
        ox  = self._map.info.origin.position.x
        oy  = self._map.info.origin.position.y
        grid = np.array(self._map.data, dtype=np.int16).reshape(h, w)

        unknown = (grid == -1)
        if not unknown.any():
            self._attract_target = None
            return None

        # Find connected clusters of unknown space
        labeled, n_clusters = scipy_label(unknown)
        best_target = None
        best_dist = float('inf')

        for cid in range(1, n_clusters + 1):
            cells = np.argwhere(labeled == cid)
            if len(cells) < self.MIN_CLUSTER_CELLS:
                continue
            cy = cells[:, 0].mean() * res + oy
            cx = cells[:, 1].mean() * res + ox
            d = math.hypot(cx - self._robot_x, cy - self._robot_y)
            if d < best_dist:
                best_dist = d
                best_target = (cx, cy)

        self._attract_target = best_target
        self._attract_update_time = now
        return best_target

    # Potential Field

    def _attractive_force(self):
        target = self._find_nearest_unknown_target()
        if target is None:
            # No unknown space found — drive forward
            return (math.cos(self._robot_yaw), math.sin(self._robot_yaw))

        cx, cy = target
        dx = cx - self._robot_x
        dy = cy - self._robot_y
        dist = math.hypot(dx, dy)
        if dist < 0.01:
            return (0.0, 0.0)

        return (self.K_ATT * dx / dist, self.K_ATT * dy / dist)

    def _repulsive_force(self):
        if self._scan is None:
            return (0.0, 0.0)

        scan = self._scan
        fx, fy = 0.0, 0.0

        for i, r in enumerate(scan.ranges):
            if math.isnan(r) or math.isinf(r):
                continue
            if r > self.D_REP or r < scan.range_min:
                continue

            beam_angle_robot = scan.angle_min + i * scan.angle_increment
            beam_angle_world = self._robot_yaw + beam_angle_robot

            if r < 0.01:
                r = 0.01
            mag = self.K_REP * (1.0/r - 1.0/self.D_REP) / (r**2)
            fx -= mag * math.cos(beam_angle_world)
            fy -= mag * math.sin(beam_angle_world)

        return (fx, fy)

    # Wall-following mode

    def _wall_follow_step(self):
        # Simple wall-following to escape local minima regions.
        if self._scan is None:
            return

        scan = self._scan
        N = len(scan.ranges)

        # Measure distance to left and right walls
        left_dist = float('inf')
        right_dist = float('inf')
        front_dist = float('inf')

        for i, r in enumerate(scan.ranges):
            if math.isnan(r) or math.isinf(r):
                continue
            angle = scan.angle_min + i * scan.angle_increment
            if abs(angle) < math.radians(30):
                front_dist = min(front_dist, r)
            elif math.radians(60) < angle < math.radians(120):
                left_dist = min(left_dist, r)
            elif math.radians(-120) < angle < math.radians(-60):
                right_dist = min(right_dist, r)

        cmd = Twist()

        # If front is blocked, turn
        if front_dist < 0.4:
            cmd.angular.z = self._wall_follow_side * self.OMEGA_MAX * 0.8
            cmd.linear.x = 0.02
        else:
            # Follow the wall on the chosen side
            if self._wall_follow_side > 0:
                wall_dist = left_dist
            else:
                wall_dist = right_dist

            # P-controller to maintain desired wall distance
            err = wall_dist - self.WALL_FOLLOW_DIST
            cmd.angular.z = -self._wall_follow_side * np.clip(err * 2.0,
                                                               -0.8, 0.8)
            cmd.linear.x = self.WALL_FOLLOW_SPEED

        self._cmd_pub.publish(cmd)

    # Control

    def _control_step(self):
        if self._scan is None:
            return

        now = time.time()

        # Wall-following mode
        if self._wall_following:
            if now < self._wall_follow_end:
                self._wall_follow_step()
                return
            else:
                self._wall_following = False
                self._consecutive_stalls = 0
                self._last_nonzero_vel = now
                self.get_logger().info('Wall-following complete — resuming potential field.')

        # Local minima escape
        stuck_for = now - self._last_nonzero_vel
        if not self._perturbing and stuck_for > self.STALL_TIME:
            self._consecutive_stalls += 1
            self.get_logger().warn(
                f'Local minimum detected (stall #{self._consecutive_stalls}) — perturbing.')

            # After repeated stalls, switch to wall-following
            if self._consecutive_stalls >= self.WALL_FOLLOW_TRIGGER:
                self.get_logger().info(
                    f'Switching to wall-following for {self.WALL_FOLLOW_DUR}s')
                self._wall_following = True
                self._wall_follow_end = now + self.WALL_FOLLOW_DUR
                # Follow the side with more space
                left_d = self._side_distance(left=True)
                right_d = self._side_distance(left=False)
                self._wall_follow_side = 1.0 if left_d < right_d else -1.0
                self._attract_target = None  # force re-evaluation
                return

            self._perturbing = True
            self._perturb_end = now + self.PERTURB_DUR
            self._perturb_dir = np.random.choice([-1.0, 1.0])

        if self._perturbing:
            if now < self._perturb_end:
                cmd = Twist()
                cmd.angular.z = self._perturb_dir * self.OMEGA_MAX * 0.7
                cmd.linear.x = 0.10  # raised from 0.05 — move forward during perturbation
                self._cmd_pub.publish(cmd)
                return
            else:
                self._perturbing = False
                self._last_nonzero_vel = now

        # Emergency brake
        front_dist = self._front_min_range()
        if front_dist < self.D_SAFE:
            cmd = Twist()
            cmd.linear.x = -0.05
            cmd.angular.z = np.random.choice([-1.0, 1.0]) * self.OMEGA_MAX
            self._cmd_pub.publish(cmd)
            return

        # Compute potential field velocity
        ax, ay = self._attractive_force()
        rx, ry = self._repulsive_force()
        fx, fy = ax + rx, ay + ry
        f_magnitude = math.hypot(fx, fy)

        if f_magnitude < 1e-4:
            self._last_nonzero_vel = 0.0
            return

        desired_yaw = math.atan2(fy, fx)
        yaw_err = desired_yaw - self._robot_yaw
        while yaw_err >  math.pi: yaw_err -= 2*math.pi
        while yaw_err < -math.pi: yaw_err += 2*math.pi

        omega = min(self.OMEGA_MAX, max(-self.OMEGA_MAX, 1.5 * yaw_err))
        speed = self.VEL_MAX * max(0.0, 1.0 - abs(yaw_err) / math.pi)
        speed = min(speed, self.VEL_MAX * min(1.0, front_dist / 0.8))

        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = omega
        self._cmd_pub.publish(cmd)

    def _front_min_range(self):
        if self._scan is None:
            return float('inf')
        scan = self._scan
        half_angle = math.radians(30)
        front_min = float('inf')
        for i, r in enumerate(scan.ranges):
            if math.isnan(r) or math.isinf(r):
                continue
            angle = scan.angle_min + i * scan.angle_increment
            if abs(angle) <= half_angle:
                front_min = min(front_min, r)
        return front_min

    def _side_distance(self, left=True):
        # Average distance on left or right side
        if self._scan is None:
            return float('inf')
        scan = self._scan
        distances = []
        for i, r in enumerate(scan.ranges):
            if math.isnan(r) or math.isinf(r):
                continue
            angle = scan.angle_min + i * scan.angle_increment
            if left and math.radians(45) < angle < math.radians(135):
                distances.append(r)
            elif not left and math.radians(-135) < angle < math.radians(-45):
                distances.append(r)
        return float(np.mean(distances)) if distances else float('inf')


def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()