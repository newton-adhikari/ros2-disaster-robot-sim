#!/usr/bin/env python3

import rclpy
from rclpy.node import Node


# explore unknown environments safely
# to make the robot move autonomously, even without a full path planner or RL agent
# it uses reactive navigation algorithm

# Attractive force → toward the centroid of unexplored space (encourages exploration).
# Repulsive force → away from obstacles detected by LiDAR (avoids collisions).

class PotentialFieldNavigator(Node):
    def __init__(self):
        super().__init__('potential_field_navigator')

        self.get_logger().info('PotentialFieldNavigator running....')


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