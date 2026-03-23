import rclpy
from rclpy.node import Node
import numpy as np


class RLNavigator(Node):
    pass

def main(args=None):
    rclpy.init(args=args)
    node = RLNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()