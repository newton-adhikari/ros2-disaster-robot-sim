#!/usr/bin/env python3

import sys
import os
import shutil
from pathlib import Path

# Path to rl_navigator.py relative to this script
THIS_DIR = Path(__file__).parent
NAVIGATOR_PATH = THIS_DIR / 'rl_navigator.py'
BACKUP_PATH    = THIS_DIR / 'rl_navigator.py.reactive_backup'

RANDOM_WALK_CODE = '''#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math
import random

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class RLNavigator(Node):
    MAX_LIN   = 0.15   # slightly slower than reactive for fair comparison
    MAX_ANG   = 0.8
    STOP_DIST = 0.35   # hard stop at walls

    def __init__(self):
        super().__init__(\'rl_navigator\')

        # Sensor state
        self.front = self.fl = self.fr = 12.0
        self.lidar_ok = False
        self.wait_ticks = 25

        # Random walk state
        self.turn_ticks   = 0
        self.turn_dir     = 1.0
        self.forward_ticks = 0

        self.create_subscription(LaserScan, \'/scan\',              self._lidar_cb, 10)
        self.create_subscription(Odometry,  \'/odometry/filtered\', self._odom_cb,  10)
        self.cmd_pub = self.create_publisher(Twist, \'/cmd_vel\', 10)
        self.create_timer(0.1, self._loop)

        self.get_logger().info(\'[RANDOM WALK BASELINE] Navigator started\')

    def _lidar_cb(self, msg):
        r = np.array(msg.ranges, dtype=np.float32)
        r = np.where(np.isfinite(r), r, msg.range_max)
        r = np.clip(r, 0.25, msg.range_max)
        n = len(r)
        inc = 2.0 * math.pi / n

        def arc(d0, d1):
            i0 = int((math.radians(d0) - msg.angle_min) / inc) % n
            i1 = int((math.radians(d1) - msg.angle_min) / inc) % n
            return float(np.min(r[i0:i1+1]) if i0 <= i1 else min(np.min(r[i0:]), np.min(r[:i1+1])))

        self.front = arc(-25,  25)
        self.fl    = arc( 25,  90)
        self.fr    = arc(-90, -25)
        self.lidar_ok = True

    def _odom_cb(self, msg):
        pass  # random walk does not use odometry

    def _loop(self):
        cmd = Twist()

        if self.wait_ticks > 0:
            self.wait_ticks -= 1
            self.cmd_pub.publish(cmd)
            return

        # If blocked — turn
        if self.front <= self.STOP_DIST or self.turn_ticks > 0:
            if self.turn_ticks <= 0:
                # Pick a random turn direction and duration
                self.turn_dir   = random.choice([-1.0, 1.0])
                self.turn_ticks = random.randint(10, 30)  # 1–3s
            self.turn_ticks -= 1
            cmd.linear.x  = 0.0
            cmd.angular.z = self.turn_dir * self.MAX_ANG
            self.cmd_pub.publish(cmd)
            return

        # Random periodic direction change (even when not blocked)
        self.forward_ticks += 1
        if self.forward_ticks > random.randint(20, 80):  # 2–8s
            self.turn_dir    = random.choice([-1.0, 1.0])
            self.turn_ticks  = random.randint(5, 20)
            self.forward_ticks = 0
            self.cmd_pub.publish(cmd)
            return

        # Drive forward at random speed
        speed = random.uniform(0.05, self.MAX_LIN)
        cmd.linear.x  = speed
        cmd.angular.z = random.uniform(-0.15, 0.15)  # slight random steering
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = RLNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == \'__main__\':
    main()
'''


def patch():
    if BACKUP_PATH.exists():
        print(f'ERROR: Backup already exists at {BACKUP_PATH}')
        print('Run "restore" first, or delete the backup manually.')
        sys.exit(1)

    if not NAVIGATOR_PATH.exists():
        print(f'ERROR: {NAVIGATOR_PATH} not found.')
        sys.exit(1)

    # Backup reactive navigator
    shutil.copy2(NAVIGATOR_PATH, BACKUP_PATH)
    print(f'Backed up reactive navigator → {BACKUP_PATH}')

    # Write random walk version
    NAVIGATOR_PATH.write_text(RANDOM_WALK_CODE)
    print(f'Patched {NAVIGATOR_PATH} with random walk baseline')


def restore():
    if not BACKUP_PATH.exists():
        sys.exit(1)

    shutil.copy2(BACKUP_PATH, NAVIGATOR_PATH)
    BACKUP_PATH.unlink()


def status():
    print(f'Navigator path: {NAVIGATOR_PATH}')
    print(f'Backup exists:  {BACKUP_PATH.exists()}')
    if NAVIGATOR_PATH.exists():
        content = NAVIGATOR_PATH.read_text()
        if 'RANDOM WALK BASELINE' in content:
            print('Current mode:   RANDOM WALK (patched)')
        else:
            print('Current mode:   REACTIVE (normal)')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: random_walk_patch.py [patch|restore|status]')
        status()
        sys.exit(0)

    cmd = sys.argv[1].lower()
    if cmd == 'patch':
        patch()
    elif cmd == 'restore':
        restore()
    elif cmd == 'status':
        status()
    else:
        print(f'Unknown command: {cmd}')
        print('Usage: random_walk_patch.py [patch|restore|status]')
        sys.exit(1)