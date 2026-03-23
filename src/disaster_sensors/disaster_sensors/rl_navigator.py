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
        cmd = Twist()

        # to pause the robot after reset 
        if self.state == 'WAIT':
            if self.wait_ticks > 0:
                self.wait_ticks -= 1
            elif self.lidar_ok and self.odom_ok:
                self.state = 'NAVIGATE' if self.mode == 'navigate' else 'FORWARD'
                self.get_logger().info(f'Ready → {self.state}')
            self.cmd_pub.publish(cmd)
            return

        # logic for turn
        if self.state == 'TURN':
            self.turn_ticks -= 1
            # Small forward arc during turn — prevents spinning in place
            cmd.linear.x  = 0.03
            cmd.angular.z = self.turn_dir * self.MAX_ANG

            if self.turn_ticks <= 0:
                if self.front > self.STOP_DIST:
                    self.state = 'FORWARD'
                    self.forward_ticks = 0
                    self.get_logger().info(f'Path clear ({self.front:.2f}m) → FORWARD')
                else:
                    # move to other direction
                    self.turn_dir *= -1
                    self.turn_ticks = 15
                    self.get_logger().info('Still blocked, trying other direction')
            self.cmd_pub.publish(cmd)
            return

        # logic for forward
        if self.state == 'FORWARD':
            self.forward_ticks += 1

            # Force a turn every 60 steps
            if self.forward_ticks > 60:
                open_left  = min(self.left, self.fl)
                open_right = min(self.right, self.fr)
                self.turn_dir   = 1.0 if open_left >= open_right else -1.0
                self.turn_ticks = 20
                self.state      = 'TURN'
                self.forward_ticks = 0
                self.get_logger().info('Periodic direction change')
                self.cmd_pub.publish(cmd)
                return

            if self.front <= self.STOP_DIST:
                self.turn_dir   = 1.0 if self.fl >= self.fr else -1.0
                self.turn_ticks = 20
                self.state      = 'TURN'
                self.forward_ticks = 0
                self.get_logger().info(
                    f'Obstacle {self.front:.2f}m → TURN '
                    f'{"left" if self.turn_dir>0 else "right"}'
                )
                cmd.linear.x  = 0.03
                cmd.angular.z = self.turn_dir * self.MAX_ANG
                self.cmd_pub.publish(cmd)
                return

            speed = self.MAX_LIN
            if self.front < self.SLOW_DIST:
                t = (self.front - self.STOP_DIST) / (self.SLOW_DIST - self.STOP_DIST)
                speed = self.MAX_LIN * max(0.3, t)

            ang = 0.0
            if self.left < 0.4 and self.left < self.right:
                ang = -0.2
            elif self.right < 0.4 and self.right < self.left:
                ang = 0.2

            cmd.linear.x  = speed
            cmd.angular.z = ang
            self.cmd_pub.publish(cmd)
            return

        # logic for navigate
        if self.state == 'NAVIGATE':
            dx = self.goal_x - self.robot_x
            dy = self.goal_y - self.robot_y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist < 0.4:
                self.get_logger().info(
                    f'Goal reached! ({self.robot_x:.1f}, {self.robot_y:.1f})')
                self.mode  = 'explore'
                self.state = 'FORWARD'
                self.forward_ticks = 0
                self.cmd_pub.publish(cmd)
                return

            # Heading error toward goal
            target = math.atan2(dy, dx)
            err    = (target - self.robot_yaw + math.pi) % (2*math.pi) - math.pi

            if self.front <= self.STOP_DIST:
                # turn toward most open side
                avoid = 1.0 if self.fl >= self.fr else -1.0
                cmd.linear.x  = 0.03
                cmd.angular.z = avoid * self.MAX_ANG
            elif abs(err) > 1.0:
                # turn in place first
                cmd.linear.x  = 0.0
                cmd.angular.z = np.clip(err * 1.2, -self.MAX_ANG, self.MAX_ANG)
            else:
                # Drive toward goal slowly
                speed = self.MAX_LIN * min(1.0, self.front / self.SLOW_DIST)
                cmd.linear.x  = max(0.05, speed)
                cmd.angular_z = np.clip(err * 0.9, -self.MAX_ANG * 0.7,
                                                     self.MAX_ANG * 0.7)
                cmd.angular.z = float(np.clip(err * 0.9, -self.MAX_ANG*0.7,
                                                           self.MAX_ANG*0.7))

            self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = RLNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()