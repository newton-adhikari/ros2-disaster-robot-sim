#!/usr/bin/env python3
import rclpy
import math, time

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Odometry


# This node is for Monitoring EKF sensor fusion quality
# wheel slips because of odom drifting
# so we see the difference between raw and filtered odom data
class EKFMonitor(Node):
    def __init__(self):
        super().__init__('ekf_monitor')

        self.log_rate = self.get_parameter('log_rate').value

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # adding subscribers
        self.raw_sub = self.create_subscription(
            Odometry, '/odom',
            self.raw_odom_callback, sensor_qos
        )
        self.ekf_sub = self.create_subscription(
            Odometry, '/odometry/filtered',
            self.ekf_odom_callback, sensor_qos
        )
        self.log_timer = self.create_timer(
            1.0 / self.log_rate, self.log_comparison
        )

    def raw_odom_callback(self, msg: Odometry):
        self.raw_odom = msg
        if self.start_time is None:
            self.start_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def ekf_odom_callback(self, msg: Odometry):
        self.ekf_odom = msg

    def log_comparison(self):
        if self.raw_odom is None or self.ekf_odom is None:
            return
        
    def destroy_node(self):
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = EKFMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n Ctrl+C —  ...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
