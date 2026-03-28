import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node

class DisasterNavEnv(gym.Env):
    def __init__(self, node: Node):
        super().__init__()
        self.node = node