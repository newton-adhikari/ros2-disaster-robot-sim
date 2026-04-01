#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

import numpy as np
from scipy.ndimage import label as scipy_label
from scipy.ndimage import binary_dilation
import math
import time


# the basic idead of this class is
# Subscribe to the occupancy grid
# Detect free cells adjacent to unknown cells
# Cluster those frontiers and select the nearest centroid
# Navigate to them using Nav2 SimpleCommander

class FrontierExplorer(Node):
    FREE_THRESH    = 25    # occupancy < 25  → free
    OCC_THRESH     = 65    # occupancy > 65  → occupied
    UNKNOWN_VAL    = -1    # occupancy == -1 → unknown
    REPLAN_DIST    = 0.3   # m — replan if goal is now too close to obstacle
    UPDATE_HZ      = 2.0   # frontier detection rate
    MIN_FRONTIER   = 5     # min frontier cluster size (cells)
    NAV_TIMEOUT    = 30.0  # s — abort goal after this duration
    GOAL_RADIUS    = 0.5   # m — consider goal reached within this radius
    STUCK_VEL_THR  = 0.01  # m/s — below this → robot is stuck
    STUCK_TIME     = 10.0  # s — if stuck this long, pick new frontier


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
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        # this is yaw from quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._robot_heading = math.atan2(siny_cosp, cosy_cosp)

        # Stuck detection
        vx = msg.twist.twist.linear.x
        wz = msg.twist.twist.angular.z
        if abs(vx) > self.STUCK_VEL_THR or abs(wz) > self.STUCK_VEL_THR:
            self._last_nonzero_vel = time.time()

    def _cancel_current_goal(self):
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
            self._goal_handle = None
        self._navigating = False

    def _detect_frontiers(self, grid: np.ndarray) -> list[tuple[float, float]]:
        
        # Returns list of (x_world, y_world) frontier centroids.

        # A frontier cell is a FREE cell with at least one UNKNOWN neighbour.
        # We cluster frontiers using connected components and return centroids
        # of clusters larger than MIN_FRONTIER cells.
        
        h, w = grid.shape

        free    = (grid >= 0) & (grid < self.FREE_THRESH)
        unknown = (grid == self.UNKNOWN_VAL)

        # Dilate unknown mask by 1 cell (4-connectivity)
        unknown_dilated = binary_dilation(unknown, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))

        # Frontier mask: free AND adjacent to unknown
        frontier_mask = free & unknown_dilated

        # Connected components
        labeled, num_features = scipy_label(frontier_mask)

        centroids = []
        map_msg = self._map
        res = map_msg.info.resolution
        ox  = map_msg.info.origin.position.x
        oy  = map_msg.info.origin.position.y

        for component_id in range(1, num_features + 1):
            cells = np.argwhere(labeled == component_id)
            if len(cells) < self.MIN_FRONTIER:
                continue
            # centroid in grid coords (row, col)
            cy_grid = cells[:, 0].mean()
            cx_grid = cells[:, 1].mean()
            # convert to world coords
            x_world = cx_grid * res + ox
            y_world = cy_grid * res + oy
            centroids.append((x_world, y_world, len(cells)))

        return centroids  # list of (x, y, size)

    def _nearest_frontier(self, centroids: list) -> tuple | None:
        # Return the nearest frontier centroid to the robot.
        if not centroids:
            return None
        dists = [math.hypot(cx - self._robot_x, cy - self._robot_y)
                 for cx, cy, _ in centroids]
        best = min(range(len(dists)), key=lambda i: dists[i])
        return centroids[best]
    
    def _explore_step(self):
        # map has to be present
        if self._map is None:
            return
        
        # Convert map to numpy
        w = self._map.info.width
        h = self._map.info.height
        grid = np.array(self._map.data, dtype=np.int16).reshape(h, w)

        # Check timeout
        if self._navigating:
            elapsed = time.time() - self._goal_sent_time
            stuck_for = time.time() - self._last_nonzero_vel
            if elapsed > self.NAV_TIMEOUT:
                self.get_logger().warn('Goal timeout. Replanning...')
                self._cancel_current_goal()
            elif stuck_for > self.STUCK_TIME:
                self.get_logger().warn('Stuck detected. Replanning...')
                self._cancel_current_goal()
            else:
                return  # still navigating

        # Detect and select frontier
        centroids = self._detect_frontiers(grid)
        if not centroids:
            self.get_logger().info('No frontiers detected — exploration complete or map not ready.')
            self._spin_in_place()
            return

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