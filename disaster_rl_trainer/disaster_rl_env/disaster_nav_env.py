import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math, time, threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

SPAWN_X, SPAWN_Y = 0.0, -2.0
GOAL_X,  GOAL_Y  = 8.0, 12.0
MAX_GOAL_DIST    = 20.0       # normalisation constant for distance obs
MAX_STEPS        = 2000   
GOAL_RADIUS      = 0.5
COLLISION_DIST   = 0.28
MAX_LIN          = 0.22
MAX_ANG          = 1.82
LIDAR_MAX        = 12.0
NUM_SECTORS      = 12
OBS_DIM          = NUM_SECTORS + 4

class DisasterNavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, node: Node):
        super().__init__()
        self.node = node

        self.observation_space = spaces.Box(
            low=np.full(OBS_DIM, -1.0, dtype=np.float32),
            high=np.ones(OBS_DIM, dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )

        self._lock   = threading.Lock()
        self.sectors = np.ones(NUM_SECTORS, dtype=np.float32) * LIDAR_MAX
        self.robot_x = self.robot_y = self.robot_yaw = 0.0
        self.vx = self.vyaw = 0.0
        self.lidar_ok = self.odom_ok = False

        # Episode state
        self.step_count      = 0
        self.prev_dist       = self._goal_dist()
        self.best_dist       = self._goal_dist()
        self.visited_cells   = set()
        self.collision_this_ep = False   
        self.collision_cooldown= 0        

        self._cmd_pub = node.create_publisher(Twist, '/cmd_vel', 10)
        node.create_subscription(LaserScan, '/scan',
                                 self._lidar_cb, rclpy.qos.qos_profile_sensor_data)
        node.create_subscription(Odometry, '/odometry/filtered',
                                 self._odom_cb, 10)
        node.get_logger().info('DisasterNavEnv v3 initialised')

    def _lidar_cb(self, msg: LaserScan):
        r = np.array(msg.ranges, dtype=np.float32)
        r = np.where(np.isfinite(r), r, msg.range_max)
        r = np.clip(r, 0.25, msg.range_max)
        n   = len(r)
        inc = 2.0 * math.pi / n

        def arc_min(d0, d1):
            i0 = int((math.radians(d0) - msg.angle_min) / inc) % n
            i1 = int((math.radians(d1) - msg.angle_min) / inc) % n
            if i0 <= i1:
                return float(np.min(r[i0:i1+1]))
            return float(min(np.min(r[i0:]), np.min(r[:i1+1])))

        bounds = [(-25,5),(5,35),(35,65),(65,95),(95,125),(125,155),
                  (155,180),(-180,-155),(-155,-125),(-125,-95),(-95,-65),(-65,-25)]
        with self._lock:
            for i,(a0,a1) in enumerate(bounds):
                self.sectors[i] = arc_min(a0, a1)
            self.lidar_ok = True

    def _odom_cb(self, msg: Odometry):
        with self._lock:
            self.robot_x  = msg.pose.pose.position.x
            self.robot_y  = msg.pose.pose.position.y
            self.vx       = msg.twist.twist.linear.x
            self.vyaw     = msg.twist.twist.angular.z
            q = msg.pose.pose.orientation
            self.robot_yaw = math.atan2(
                2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
            self.odom_ok = True

    def _goal_dist(self):
        return math.sqrt((GOAL_X-self.robot_x)**2 + (GOAL_Y-self.robot_y)**2)

    def _goal_bearing(self):
        dx, dy = GOAL_X-self.robot_x, GOAL_Y-self.robot_y
        b = math.atan2(dy,dx) - self.robot_yaw
        return (b + math.pi) % (2*math.pi) - math.pi

    def _obs(self):
        with self._lock:
            sectors = self.sectors.copy()
            vx, vyaw = self.vx, self.vyaw
        dist    = self._goal_dist()
        bearing = self._goal_bearing()
        return np.array([
            *sectors / LIDAR_MAX,
            np.clip(1.0 - dist / MAX_GOAL_DIST, -1.0, 1.0),
            bearing / math.pi,
            vx   / MAX_LIN,
            vyaw / MAX_ANG,
        ], dtype=np.float32)


    # gym api implement
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._stop()
        time.sleep(0.3)

        self.step_count        = 0
        self.visited_cells     = set()
        self.prev_dist         = self._goal_dist()
        self.best_dist         = self._goal_dist()
        self.collision_this_ep = False
        self.collision_cooldown= 0

        timeout = time.time() + 3.0
        while not (self.lidar_ok and self.odom_ok) and time.time() < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.05)

        return self._obs(), {}

    def step(self, action):
        lin = float(np.clip(action[0] * MAX_LIN, -MAX_LIN, MAX_LIN))
        ang = float(np.clip(action[1] * MAX_ANG, -MAX_ANG, MAX_ANG))

        cmd = Twist()
        cmd.linear.x  = lin
        cmd.angular.z = ang
        self._cmd_pub.publish(cmd)

        rclpy.spin_once(self.node, timeout_sec=0.05)
        time.sleep(0.05)

        self.step_count += 1
        if self.collision_cooldown > 0:
            self.collision_cooldown -= 1

        obs      = self._obs()
        dist     = self._goal_dist()
        min_lidar= float(np.min(self.sectors))

        bearing = self._goal_bearing()  # radians, 0 = facing goal

        # 1. Heading alignment — to teach the bot "face the goal"
        #    cos(bearing)=1.0 when facing goal, -1.0 when facing away
        alignment = math.cos(bearing)
        reward += alignment * 1.0   # +1.0/step facing goal, -1.0 facing away

        # 2. Progress reward — moving closer to goal
        progress = self.prev_dist - dist
        reward  += progress * 80.0   # was 40 — doubled

        # 3. Forward-toward-goal bonus — only reward forward motion when aligned
        if self.vx > 0.04 and alignment > 0.5:
            reward += 0.5

        # 4. Best-ever distance bonus
        if dist < self.best_dist - 0.1:
            reward += (self.best_dist - dist) * 20.0
            self.best_dist = dist

        # 5. Goal reached
        goal_reached = dist < GOAL_RADIUS
        if goal_reached:
            reward += 500.0

        # 6. Collision
        is_collision = min_lidar < COLLISION_DIST and self.collision_cooldown == 0
        if is_collision:
            reward -= 30.0
            self.collision_cooldown = 20

        # 7. Coverage novelty — reduced, exploration is secondary to goal-seeking
        cx = round(self.robot_x * 2) / 2
        cy = round(self.robot_y * 2) / 2
        key = (cx, cy)
        if key not in self.visited_cells:
            self.visited_cells.add(key)
            reward += 1.0    # was 5.0 — reduced so it doesn't dominate

        if dist < 5.0:
            reward += (5.0 - dist) * 2.0

        # Time penalty 
        reward -= 0.01

        terminated = goal_reached
        truncated  = self.step_count >= MAX_STEPS

        if terminated or truncated:
            self._stop()

        self.prev_dist = dist

        info = {
            "dist_to_goal":   dist,
            "goal_reached":   goal_reached,
            "collision":      is_collision,
            "coverage_cells": len(self.visited_cells),
            "step":           self.step_count,
        }
        return obs, reward, terminated, truncated, info

    def _stop(self):
        self._cmd_pub.publish(Twist())

    def close(self):
        self._stop()
