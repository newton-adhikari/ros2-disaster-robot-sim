#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist

import numpy as np
import math
import time


# explore unknown environments safely
# to make the robot move autonomously, even without a full path planner or RL agent
# it uses reactive navigation algorithm

# Attractive force → toward the centroid of unexplored space (encourages exploration).
# Repulsive force → away from obstacles detected by LiDAR (avoids collisions).

class PotentialFieldNavigator(Node):
    
    STALL_VEL   = 0.02    # m/s — below this = stall
    UPDATE_HZ   = 10.0

    def __init__(self):
        super().__init__('potential_field_navigator')

        self._scan: LaserScan | None = None
        self._map: OccupancyGrid | None = None
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_yaw = 0.0
        self._last_nonzero_vel = time.time()
        self._perturbing = False
        self._perturb_end = 0.0
        self._perturb_dir = 1.0

        # QoS
        latch_qos = QoSProfile(depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        best_effort_qos = QoSProfile(depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT)

        self.create_subscription(LaserScan, '/scan', self._scan_cb, best_effort_qos)
        self.create_subscription(OccupancyGrid,'/map', self._map_cb, latch_qos)
        self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb, best_effort_qos)

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(1.0 / self.UPDATE_HZ, self._control_step)

        self.get_logger().info('PotentialFieldNavigator running....')

    def _scan_cb(self, msg: LaserScan):
        self._scan = msg

    def _map_cb(self, msg: OccupancyGrid):
        self._map = msg

    def _odom_cb(self, msg: Odometry):
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

    # fields to implement the forces
    def _attractive_force(self) -> tuple[float, float]:
        # the attractive_force :  Force toward the centroid of unknown space.
        if self._map is None:
            # default: forward
            return (math.cos(self._robot_yaw), math.sin(self._robot_yaw))

        w = self._map.info.width
        h = self._map.info.height
        res = self._map.info.resolution
        ox  = self._map.info.origin.position.x
        oy  = self._map.info.origin.position.y
        grid = np.array(self._map.data, dtype=np.int16).reshape(h, w)

        unknown_mask = (grid == -1)
        if not unknown_mask.any():
            return (0.0, 0.0)

        ys, xs = np.where(unknown_mask)
        # centroid of unknown space in world coords
        cx = xs.mean() * res + ox
        cy = ys.mean() * res + oy

        dx = cx - self._robot_x
        dy = cy - self._robot_y
        dist = math.hypot(dx, dy)
        if dist < 0.01:
            return (0.0, 0.0)

        # normalise (linear attractive potential)
        return (self.K_ATT * dx / dist, self.K_ATT * dy / dist)

    def _repulsive_force(self) -> tuple[float, float]:
        # Repulsive force from obstacles in LiDAR scan.
        if self._scan is None:
            return (0.0, 0.0)

        scan = self._scan
        N = len(scan.ranges)
        fx, fy = 0.0, 0.0

        for i, r in enumerate(scan.ranges):
            if math.isnan(r) or math.isinf(r):
                continue
            if r > self.D_REP or r < scan.range_min:
                continue

            # Physical angle of this beam in world frame
            beam_angle_robot = scan.angle_min + i * scan.angle_increment
            beam_angle_world = self._robot_yaw + beam_angle_robot

            # Repulsive magnitude: gradient of 0.5 * K * (1/d - 1/d0)^2
            if r < 0.01:
                r = 0.01
            mag = self.K_REP * (1.0/r - 1.0/self.D_REP) / (r**2)
            # Direction: away from obstacle
            fx -= mag * math.cos(beam_angle_world)
            fy -= mag * math.sin(beam_angle_world)

        return (fx, fy)

    def _total_force(self) -> tuple[float, float]:
        ax, ay = self._attractive_force()
        rx, ry = self._repulsive_force()
        return (ax + rx, ay + ry)


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