import rclpy
from rclpy.node import Node

class EKFDataCollector(Node):

    def __init__(self):
        super().__init__('ekf_data_collector')


def main():
    rclpy.init()
    node = EKFDataCollector()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()