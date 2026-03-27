#!/usr/bin/env python3
import rclpy
import math
import csv

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Odometry
from datetime import datetime


# This node is for Monitoring EKF sensor fusion quality
# wheel slips because of odom drifting
# so we see the difference between raw and filtered odom data
class EKFMonitor(Node):
    
    def __init__(self):
        super().__init__('ekf_monitor')

        # Parameters
        self.declare_parameter('log_csv',    True)
        self.declare_parameter('log_rate',   1.0)    # Hz — how often to log

        self.log_csv  = self.get_parameter('log_csv').value
        self.log_rate = self.get_parameter('log_rate').value

        # State
        self.raw_odom = None
        self.ekf_odom = None
        self.start_time = None

        # Running error tracking
        self.position_errors = []
        self.covariance_traces = []
        self.sample_count = 0

        if self.log_csv:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.csv_path = f'/tmp/ekf_comparison_{ts}.csv'
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                'time_s',
                'raw_x', 'raw_y', 'raw_yaw',
                'ekf_x', 'ekf_y', 'ekf_yaw',
                'pos_error_m',
                'yaw_error_rad',
                'ekf_cov_trace',
                'ekf_cov_xx', 'ekf_cov_yy', 'ekf_cov_yaw'
            ])
            self.get_logger().info(f'Logging EKF comparison to: {self.csv_path}')

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

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

        self.get_logger().info('EKFMonitor started for /odom and /odometry/filtered...')

    @staticmethod
    def quat_to_yaw(q):
        # Extracting yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def raw_odom_callback(self, msg: Odometry):
        self.raw_odom = msg
        if self.start_time is None:
            self.start_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def ekf_odom_callback(self, msg: Odometry):
        self.ekf_odom = msg

    def log_comparison(self):
        if self.raw_odom is None or self.ekf_odom is None:
            return

        # Extracting positions
        rx = self.raw_odom.pose.pose.position.x
        ry = self.raw_odom.pose.pose.position.y
        ry_raw = self.quat_to_yaw(self.raw_odom.pose.pose.orientation)

        ex = self.ekf_odom.pose.pose.position.x
        ey = self.ekf_odom.pose.pose.position.y
        ey_ekf = self.quat_to_yaw(self.ekf_odom.pose.pose.orientation)

        # Position divergence between raw odom and EKF
        pos_error = math.sqrt((rx - ex) ** 2 + (ry - ey) ** 2)

        # Yaw divergence
        yaw_error = abs(ry_raw - ey_ekf)
        if yaw_error > math.pi:
            yaw_error = 2 * math.pi - yaw_error

        # EKF covariance trace 
        cov = self.ekf_odom.pose.covariance  # 6×6 matrix
        cov_xx  = cov[0]   # x uncertainty
        cov_yy  = cov[7]   # y uncertainty
        cov_yaw = cov[35]  # yaw uncertainty
        cov_trace = cov_xx + cov_yy + cov_yaw

        # Running stats
        self.position_errors.append(pos_error)
        self.covariance_traces.append(cov_trace)
        self.sample_count += 1

        # Timestamp
        t_now = self.raw_odom.header.stamp.sec + self.raw_odom.header.stamp.nanosec * 1e-9
        t_rel = t_now - self.start_time if self.start_time else 0.0

        # Log to CSV
        if self.log_csv:
            self.csv_writer.writerow([
                f'{t_rel:.3f}',
                f'{rx:.4f}', f'{ry:.4f}', f'{ry_raw:.4f}',
                f'{ex:.4f}', f'{ey:.4f}', f'{ey_ekf:.4f}',
                f'{pos_error:.4f}',
                f'{yaw_error:.4f}',
                f'{cov_trace:.6f}',
                f'{cov_xx:.6f}', f'{cov_yy:.6f}', f'{cov_yaw:.6f}',
            ])
            self.csv_file.flush()

        # Console log every 10 samples
        if self.sample_count % 10 == 0:
            mean_err = sum(self.position_errors) / len(self.position_errors)
            self.get_logger().info(
                f'[EKF Monitor #{self.sample_count}] '
                f'pos_divergence={pos_error:.3f}m | '
                f'yaw_divergence={math.degrees(yaw_error):.1f}° | '
                f'cov_trace={cov_trace:.4f} | '
                f'mean_pos_error={mean_err:.3f}m'
            )

        # Summary every 60 samples (~1 min at 1Hz)
        if self.sample_count % 60 == 0:
            mean_err = sum(self.position_errors) / len(self.position_errors)
            rmse = math.sqrt(sum(e**2 for e in self.position_errors) / len(self.position_errors))
            self.get_logger().info(
                f'  Mean EKF position divergence: {mean_err:.4f} m\n'
                f'  Position RMSE (odom vs EKF): {rmse:.4f} m\n'
            )

    def destroy_node(self):
        if self.log_csv and hasattr(self, 'csv_file'):
            # Write final summary as last row comment
            if self.position_errors:
                mean_err = sum(self.position_errors) / len(self.position_errors)
                rmse = math.sqrt(
                    sum(e**2 for e in self.position_errors) / len(self.position_errors)
                )
                self.csv_writer.writerow([
                    '# SUMMARY',
                    f'samples={self.sample_count}',
                    f'mean_pos_divergence={mean_err:.4f}m',
                    f'rmse={rmse:.4f}m',
                    '', '', '', '', '', '', '', '', ''
                ])
            self.csv_file.close()
            self.get_logger().info(f'CSV saved: {self.csv_path}')
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
