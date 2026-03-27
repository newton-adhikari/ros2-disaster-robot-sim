import rclpy
from rclpy.node import Node
import time

from nav_msgs.msg import Odometry


class EKFDataCollector(Node):

    def __init__(self, duration: float):
        super().__init__('ekf_data_collector')
        self.duration   = duration
        self.raw_odom   = [] 
        self.ekf_odom   = [] 
        self.groundtruth= [] 
        self.start_time = None

        # using this to match robot_localization publisher
        qos = 10

        self.create_subscription(Odometry, '/odom', self._raw_cb, qos)

    def _raw_cb(self, msg):
        t = self._ts(msg.header)

        if self.start_time is None:
            self.start_time = time.time()
            self.get_logger().info('Recording started — please have the robot running!')
        
        # record the data
        self.raw_odom.append((t, msg.pose.pose.position.x, msg.pose.pose.position.y, self._yaw(msg.pose.pose.orientation)))


def main():
    rclpy.init()
    node = EKFDataCollector()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()