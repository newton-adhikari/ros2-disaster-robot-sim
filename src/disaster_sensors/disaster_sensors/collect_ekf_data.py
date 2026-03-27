import rclpy
from rclpy.node import Node
import time

from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates


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
        self.create_subscription(Odometry, '/odometry/filtered', self._ekf_cb, qos)
        self.create_subscription(ModelStates, '/gazebo/model_states', self._gt_cb, qos)

    def _raw_cb(self, msg):
        t = self._ts(msg.header)

        if self.start_time is None:
            self.start_time = time.time()
            self.get_logger().info('Recording started — please have the robot running!')
        
        # record the data
        self.raw_odom.append((t, msg.pose.pose.position.x, msg.pose.pose.position.y, self._yaw(msg.pose.pose.orientation)))

    # helper method for timestamp
    def _ts(self, header):
        return header.stamp.sec + header.stamp.nanosec * 1e-9
    
    def _ekf_cb(self, msg):
        if self.start_time is None:
            return
        t = self._ts(msg.header)

        # filtered odom data
        self.ekf_odom.append((t, msg.pose.pose.position.x, msg.pose.pose.position.y, self._yaw(msg.pose.pose.orientation)))

    def _gt_cb(self, msg):
        if self.start_time is None:
            return
        try:
            idx = msg.name.index('disaster_robot')
        except ValueError:
            return
        p = msg.pose[idx]
        
        self.groundtruth.append((time.time(), p.position.x, p.position.y, self._yaw(p.orientation)))


def main():
    rclpy.init()
    node = EKFDataCollector()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()