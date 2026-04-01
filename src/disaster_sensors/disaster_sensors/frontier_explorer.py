#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

import math
import time


# the basic idead of this class is
# Subscribe to the occupancy grid
# Detect free cells adjacent to unknown cells
# Cluster those frontiers and select the nearest centroid
# Navigate to them using Nav2 SimpleCommander

class FrontierExplorer(Node):
    UPDATE_HZ      = 2.0   # frontier detection rate
    MIN_FRONTIER   = 5     # min frontier cluster size (cells)
    NAV_TIMEOUT    = 30.0  # s — abort goal after this duration
    GOAL_RADIUS    = 0.5   # m — consider goal reached within this radius

    def __init__(self):
        super().__init__('frontier_explorer')
        # Parameters
        self.declare_parameter('update_hz', self.UPDATE_HZ)
        self.declare_parameter('min_frontier_cells', self.MIN_FRONTIER)
        self.declare_parameter('nav_timeout', self.NAV_TIMEOUT)
        self.declare_parameter('goal_radius', self.GOAL_RADIUS)
        hz = self.get_parameter('update_hz').value

        self._map: OccupancyGrid | None = None
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_heading = 0.0
        self._current_goal: PoseStamped | None = None
        self._goal_sent_time = 0.0
        self._navigating = False
        self._last_vel_time = time.time()
        self._last_nonzero_vel = time.time()
        self._goal_handle = None

        # Nav2 action client
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        odom_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

                # Subscriptions
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, map_qos)
        self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb, odom_qos)

        # Cmd_vel publisher (for fallback when Nav2 unavailable)
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Main loop timer
        self.create_timer(1.0 / hz, self._explore_step)

        self.get_logger().info('FrontierExplorer initialised. Waiting for map...')


        self.get_logger().info('FrontierExplorer initialised. Waiting for map...')

    def _map_cb(self, msg: OccupancyGrid):
        self._map = msg

    def _odom_cb(self, msg: Odometry):
        pass

    def _explore_step(self):
        pass

def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()