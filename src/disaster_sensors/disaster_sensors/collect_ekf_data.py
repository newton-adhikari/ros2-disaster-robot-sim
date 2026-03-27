import rclpy
import math, time, argparse

from rclpy.node import Node
from pathlib import Path
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

        self.get_logger().info(f'Recording for {duration}s ')
        self.get_logger().info('Waiting for /odom and /odometry/filtered ...')

    @staticmethod
    def _yaw(q):
        return math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

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

    def compute_and_save(self, out_dir: Path):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=300.0)
    parser.add_argument('--output', type=str, default=str(Path.home()))
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = EKFDataCollector(duration=args.duration)

    print(f'\nRecording for {args.duration:.0f}s ...')

    last_print = 0

    try:
        while rclpy.ok() and not node.is_done():
            rclpy.spin_once(node, timeout_sec=0.05)
            elapsed = time.time() - (node.start_time or time.time())
            if node.start_time and elapsed - last_print >= 30:
                last_print = elapsed
                print(f'  t={elapsed:.0f}s  raw={len(node.raw_odom)}  '
                      f'ekf={len(node.ekf_odom)}  gt={len(node.groundtruth)}')
    except KeyboardInterrupt:
        print('\nCtrl+C — computing with data collected so far ...')

    node.compute_and_save(out_dir)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()