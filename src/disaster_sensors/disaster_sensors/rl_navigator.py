import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math


class RLNavigator(Node):
    def __init__(self):
        super().__init__('rl_navigator')

        self.declare_parameter('mode', 'explore')
        self.declare_parameter('goal_x', 8.0)
        self.declare_parameter('goal_y', 12.0)
        self.declare_parameter('model_path', '')

        self.mode   = self.get_parameter('mode').value
        self.goal_x = float(self.get_parameter('goal_x').value)
        self.goal_y = float(self.get_parameter('goal_y').value)

        # Sensor state
        self.front = self.fl = self.fr = self.left = self.right = 12.0
        self.robot_x = self.robot_y = self.robot_yaw = 0.0
        self.lidar_ok = self.odom_ok = False

        self.state       = 'WAIT'
        self.turn_dir    = 1.0
        self.turn_ticks  = 0
        self.forward_ticks = 0   
        self.wait_ticks  = 25  

        self.create_subscription(LaserScan, '/scan', self._lidar_cb, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb,  10)
        self.create_subscription(PoseStamped, '/goal_pose', self._goal_cb,  10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(0.1, self._loop)  

        self.get_logger().info('RL Navigator agent started')

    def _lidar_cb(self, msg):
        r = np.array(msg.ranges, dtype=np.float32)
        r = np.where(np.isfinite(r), r, msg.range_max)
        r = np.clip(r, 0.25, msg.range_max)   # 0.25m filter removes self-hits

        n   = len(r)
        inc = 2.0 * math.pi / n

        def arc(d0, d1):
            i0 = int((math.radians(d0) - msg.angle_min) / inc) % n
            i1 = int((math.radians(d1) - msg.angle_min) / inc) % n
            return float(np.min(r[i0:i1+1]) if i0 <= i1
                         else min(np.min(r[i0:]), np.min(r[:i1+1])))

        self.front = arc(-25,  25)
        self.fl    = arc( 25,  90)
        self.fr    = arc(-90, -25)
        self.left  = arc( 60, 120)
        self.right = arc(-120, -60)
        self.lidar_ok = True

    def _odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(
            2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        self.odom_ok = True

    def _goal_cb(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.mode   = 'navigate'
        self.state  = 'NAVIGATE'
        self.get_logger().info(f'Goal: ({self.goal_x:.1f}, {self.goal_y:.1f})')

    def _loop(self):
        pass
        # will implement later

def main(args=None):
    rclpy.init(args=args)
    node = RLNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()