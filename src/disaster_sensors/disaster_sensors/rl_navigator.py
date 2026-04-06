#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math
import time
import os

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

# RL constants — from disaster_nav_env.py
LIDAR_MAX    = 12.0
NUM_SECTORS  = 12
OBS_DIM      = NUM_SECTORS + 4
MAX_LIN_RL   = 0.22
MAX_ANG_RL   = 1.82
GOAL_X, GOAL_Y = 8.0, 12.0


class RLNavigator(Node):

    MAX_LIN    = 0.18   # m/s (FSM mode)
    MAX_ANG    = 0.8    # rad/s
    STOP_DIST  = 0.35   # m — start turning
    SLOW_DIST  = 0.70   # m — start slowing

    def __init__(self):
        super().__init__('rl_navigator')

        self.declare_parameter('mode',       'explore')
        self.declare_parameter('goal_x',      8.0)
        self.declare_parameter('goal_y',     12.0)
        self.declare_parameter('model_path',  '')

        self.mode       = self.get_parameter('mode').value
        self.goal_x     = float(self.get_parameter('goal_x').value)
        self.goal_y     = float(self.get_parameter('goal_y').value)
        model_path_str  = self.get_parameter('model_path').value

        # RL model loading
        self.rl_model = None
        self.use_rl   = False

        if model_path_str and os.path.isfile(model_path_str):
            try:
                from stable_baselines3 import PPO
                # SB3 .zip model
                self.rl_model = PPO.load(model_path_str, device='cpu')
                self.use_rl = True
                self.get_logger().info(
                    f'RL model loaded: {model_path_str}')
            except Exception as e:
                self.get_logger().error(
                    f'Failed to load RL model: {e}. Falling back to FSM.')
        elif model_path_str:
            self.get_logger().warn(
                f'Model file not found: {model_path_str}. Using FSM.')

        # Sensor state 
        self.front = self.fl = self.fr = self.left = self.right = 12.0
        self.robot_x = self.robot_y = self.robot_yaw = 0.0
        self.vx = self.vyaw = 0.0
        self.lidar_ok = self.odom_ok = False

        # LiDAR sectors
        self.sectors = np.ones(NUM_SECTORS, dtype=np.float32) * LIDAR_MAX

        # State machine (FSM mode)
        self.state       = 'WAIT'
        self.turn_dir    = 1.0
        self.turn_ticks  = 0
        self.forward_ticks = 0
        self.wait_ticks  = 25    # 2.5s warmup

        # v2: no-progress escape
        self.last_x = 0.0
        self.last_y = 0.0
        self.progress_check_time = time.time()
        self.NO_PROGRESS_S = 30.0
        self.NO_PROGRESS_DIST_M = 0.5
        self.escape_ticks = 0

        self.create_subscription(LaserScan,   '/scan',      self._lidar_cb, 10)
        self.create_subscription(Odometry,    '/odom',      self._odom_cb,  10)
        self.create_subscription(PoseStamped, '/goal_pose', self._goal_cb,  10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(0.1, self._loop)

        mode_str = 'RL policy' if self.use_rl else 'Reactive FSM'
        self.get_logger().info(f'RL Navigator started | mode={mode_str}')

    # Callbacks

    def _lidar_cb(self, msg):
        r = np.array(msg.ranges, dtype=np.float32)
        r = np.where(np.isfinite(r), r, msg.range_max)
        r = np.clip(r, 0.25, msg.range_max)

        n   = len(r)
        inc = 2.0 * math.pi / n

        def arc(d0, d1):
            i0 = int((math.radians(d0) - msg.angle_min) / inc) % n
            i1 = int((math.radians(d1) - msg.angle_min) / inc) % n
            return float(np.min(r[i0:i1+1]) if i0 <= i1
                         else min(np.min(r[i0:]), np.min(r[:i1+1])))

        # 5-zone summary for FSM
        self.front = arc(-25,  25)
        self.fl    = arc( 25,  90)
        self.fr    = arc(-90, -25)
        self.left  = arc( 60, 120)
        self.right = arc(-120, -60)

        # 12-sector observation for RL
        bounds = [(-25,5),(5,35),(35,65),(65,95),(95,125),(125,155),
                  (155,180),(-180,-155),(-155,-125),(-125,-95),(-95,-65),(-65,-25)]
        for i, (a0, a1) in enumerate(bounds):
            self.sectors[i] = arc(a0, a1)

        self.lidar_ok = True

    def _odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.vx      = msg.twist.twist.linear.x
        self.vyaw    = msg.twist.twist.angular.z
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

    #  RL observation
    def _rl_obs(self):
        dist = math.sqrt((GOAL_X - self.robot_x)**2 + (GOAL_Y - self.robot_y)**2)
        dx, dy = GOAL_X - self.robot_x, GOAL_Y - self.robot_y
        bearing = math.atan2(dy, dx) - self.robot_yaw
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        return np.array([
            *self.sectors / LIDAR_MAX,
            1.0 / (1.0 + dist),
            bearing / math.pi,
            self.vx   / MAX_LIN_RL,
            self.vyaw / MAX_ANG_RL,
        ], dtype=np.float32)

    # Main loop

    def _loop(self):
        cmd = Twist()

        # WAIT (both modes)
        if self.state == 'WAIT':
            if self.wait_ticks > 0:
                self.wait_ticks -= 1
            elif self.lidar_ok and self.odom_ok:
                if self.use_rl:
                    self.state = 'RL'
                    self.get_logger().info('Sensors ready → RL policy active')
                else:
                    self.state = 'NAVIGATE' if self.mode == 'navigate' else 'FORWARD'
                    self.get_logger().info(f'Ready → {self.state}')
            self.cmd_pub.publish(cmd)
            return

        # RL POLICY MODE
        if self.state == 'RL' and self.use_rl:
            obs = self._rl_obs()
            action, _ = self.rl_model.predict(obs, deterministic=True)
            cmd.linear.x  = float(np.clip(action[0] * MAX_LIN_RL, -MAX_LIN_RL, MAX_LIN_RL))
            cmd.angular.z = float(np.clip(action[1] * MAX_ANG_RL, -MAX_ANG_RL, MAX_ANG_RL))

            # Safety override — emergency stop if about to collide
            if self.front < 0.25:
                cmd.linear.x = min(cmd.linear.x, 0.0)

            self.cmd_pub.publish(cmd)
            return

        # NO-PROGRESS ESCAPE
        if self.escape_ticks > 0:
            self.escape_ticks -= 1
            cmd.linear.x  = 0.10
            cmd.angular.z = self.turn_dir * self.MAX_ANG * 0.8
            self.cmd_pub.publish(cmd)
            return

        now = time.time()
        if now - self.progress_check_time > self.NO_PROGRESS_S:
            dist_moved = math.sqrt(
                (self.robot_x - self.last_x)**2 +
                (self.robot_y - self.last_y)**2)
            if dist_moved < self.NO_PROGRESS_DIST_M:
                self.get_logger().warn(
                    f'No progress ({dist_moved:.2f}m in {self.NO_PROGRESS_S}s) '
                    f'— escape manoeuvre')
                self.turn_dir = np.random.choice([-1.0, 1.0])
                self.escape_ticks = 40
            self.last_x = self.robot_x
            self.last_y = self.robot_y
            self.progress_check_time = now

        # TURN
        if self.state == 'TURN':
            self.turn_ticks -= 1
            cmd.linear.x  = 0.03
            cmd.angular.z = self.turn_dir * self.MAX_ANG

            if self.turn_ticks <= 0:
                if self.front > self.STOP_DIST:
                    self.state = 'FORWARD'
                    self.forward_ticks = 0
                    self.get_logger().info(f'Path clear ({self.front:.2f}m) → FORWARD')
                else:
                    self.turn_dir *= -1
                    self.turn_ticks = 15
                    self.get_logger().info('Still blocked, trying other direction')
            self.cmd_pub.publish(cmd)
            return

        # FORWARD (explore)
        if self.state == 'FORWARD':
            self.forward_ticks += 1

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
                    f'{"left" if self.turn_dir>0 else "right"}')
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

        # NAVIGATE
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

            target = math.atan2(dy, dx)
            err    = (target - self.robot_yaw + math.pi) % (2*math.pi) - math.pi

            if self.front <= self.STOP_DIST:
                avoid = 1.0 if self.fl >= self.fr else -1.0
                cmd.linear.x  = 0.03
                cmd.angular.z = avoid * self.MAX_ANG
            elif abs(err) > 1.0:
                cmd.linear.x  = 0.0
                cmd.angular.z = np.clip(err * 1.2, -self.MAX_ANG, self.MAX_ANG)
            else:
                speed = self.MAX_LIN * min(1.0, self.front / self.SLOW_DIST)
                cmd.linear.x  = max(0.05, speed)
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
