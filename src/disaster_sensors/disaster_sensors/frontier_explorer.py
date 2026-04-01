#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        self.get_logger().info('FrontierExplorer initialised. Waiting for map...')


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