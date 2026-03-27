#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Float32MultiArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


class LidarProcessor(Node):

    NUM_SECTORS       = 8
    GAP_MIN_WIDTH_RAD = 0.35    # ~20° minimum gap width to be navigable
    GAP_MIN_DIST_M    = 0.50    # gap must be at least this far (0.5m)
    OBSTACLE_WARN_M   = 0.35    # warn if obstacle closer than this

    def __init__(self):
        super().__init__('lidar_processor')

        self.declare_parameter('scan_topic',   '/scan')
        self.declare_parameter('min_range_use', 0.12)   # ignore <12cm (robot body)
        self.declare_parameter('max_range_use', 10.0)   # ignore >10m (too far)

        scan_topic    = self.get_parameter('scan_topic').value
        self.min_r    = self.get_parameter('min_range_use').value
        self.max_r    = self.get_parameter('max_range_use').value

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, sensor_qos
        )

        self.nearest_pub = self.create_publisher(Float32, '/scan/nearest_obstacle', 10)
        self.gaps_pub    = self.create_publisher(Float32MultiArray, '/scan/gap_angles',       10)
        self.sector_pub  = self.create_publisher(Float32MultiArray, '/scan/sector_stats',     10)

        self.scan_count      = 0
        self.total_gaps_seen = 0

        self.get_logger().info(
            f'LidarProcessor started | topic={scan_topic} | '
            f'range=[{self.min_r},{self.max_r}]m | sectors={self.NUM_SECTORS}'
        )


    def _clean_ranges(self, msg: LaserScan):
        # to Replace invalid readings with max_range
        ranges = []
        for r in msg.ranges:
            if math.isnan(r) or math.isinf(r) or r < self.min_r:
                ranges.append(self.max_r)
            elif r > self.max_r:
                ranges.append(self.max_r)
            else:
                ranges.append(r)
        return ranges

    def _compute_sector_stats(self, ranges, n):
        sector_size = n // self.NUM_SECTORS
        stats = []
        for s in range(self.NUM_SECTORS):
            start = s * sector_size
            end   = start + sector_size
            sector = ranges[start:end]
            min_d  = min(sector)
            mean_d = sum(sector) / len(sector)
            stats.append((min_d, mean_d))
        return stats

    def _detect_gaps(self, ranges, angle_min, angle_increment):
        in_gap     = False
        gap_start  = 0
        gap_angles = []

        for i, r in enumerate(ranges):
            if r >= self.GAP_MIN_DIST_M:
                if not in_gap:
                    in_gap    = True
                    gap_start = i
            else:
                if in_gap:
                    gap_end   = i - 1
                    gap_width = (gap_end - gap_start) * angle_increment
                    if gap_width >= self.GAP_MIN_WIDTH_RAD:
                        centre_idx   = (gap_start + gap_end) // 2
                        centre_angle = angle_min + centre_idx * angle_increment
                        gap_angles.append(float(centre_angle))
                    in_gap = False

        # Handle gap wrapping to end of scan
        if in_gap:
            gap_end   = len(ranges) - 1
            gap_width = (gap_end - gap_start) * angle_increment
            if gap_width >= self.GAP_MIN_WIDTH_RAD:
                centre_idx   = (gap_start + gap_end) // 2
                centre_angle = angle_min + centre_idx * angle_increment
                gap_angles.append(float(centre_angle))

        return gap_angles

    def scan_callback(self, msg: LaserScan):
        self.scan_count += 1
        ranges = self._clean_ranges(msg)
        n = len(ranges)

        # 1. Nearest obstacle
        nearest = min(ranges)
        nearest_msg = Float32()
        nearest_msg.data = float(nearest)
        self.nearest_pub.publish(nearest_msg)

        if nearest < self.OBSTACLE_WARN_M:
            self.get_logger().warn(
                f'Obstacle at {nearest:.2f}m (below {self.OBSTACLE_WARN_M}m threshold!)',
                throttle_duration_sec=1.0
            )

        # 2. Sector statistics
        stats = self._compute_sector_stats(ranges, n)
        sector_msg = Float32MultiArray()
        # Pack as [min0, mean0, min1, mean1, ...] — 2 values per sector
        sector_msg.data = [val for pair in stats for val in pair]
        self.sector_pub.publish(sector_msg)

        # 3. Gap detection
        gaps = self._detect_gaps(ranges, msg.angle_min, msg.angle_increment)
        self.total_gaps_seen += len(gaps)
        gaps_msg = Float32MultiArray()
        gaps_msg.data = gaps if gaps else [-999.0]  # -999 = no gaps found
        self.gaps_pub.publish(gaps_msg)

        # Log summary every 50 scans
        if self.scan_count % 50 == 0:
            sector_mins = [f'{s[0]:.2f}' for s in stats]
            self.get_logger().info(
                f'Scan #{self.scan_count} | '
                f'nearest={nearest:.2f}m | '
                f'gaps={len(gaps)} | '
                f'sector_mins=[{", ".join(sector_mins)}]'
            )


def main(args=None):
    rclpy.init(args=args)
    node = LidarProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f'Shutting down | processed {node.scan_count} scans | '
            f'detected {node.total_gaps_seen} gaps total'
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()