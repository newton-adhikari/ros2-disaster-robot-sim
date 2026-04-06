
import os
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from PIL import Image

class AutoMapSaver(Node):

    def __init__(self):
        super().__init__('auto_map_saver')

        self.declare_parameter('output_dir', os.path.expanduser('~/disaster-lab/disaster_results'))
        self.declare_parameter('output_prefix', 'map')
        self.declare_parameter('save_interval', 30.0)

        self.output_dir = self.get_parameter('output_dir').value
        self.output_prefix = self.get_parameter('output_prefix').value
        self.save_interval = self.get_parameter('save_interval').value

        os.makedirs(self.output_dir, exist_ok=True)

        self._last_map = None
        self._map_count = 0

        # Subscribe to /map with transient local durability (latched)
        map_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, map_qos)
        self.create_timer(self.save_interval, self._save_timer)

        self.get_logger().info(
            f'AutoMapSaver ready. Saving to {self.output_dir}/{self.output_prefix} '
            f'every {self.save_interval}s')

    def _map_cb(self, msg):
        self._last_map = msg
        self._map_count += 1
        if self._map_count == 1:
            self.get_logger().info('Received first /map message.')

    def _save_timer(self):
        if self._last_map is None:
            self.get_logger().info('No map received yet, skipping save.')
            return
        self._save_map(self._last_map)

    def _save_map(self, grid):
        prefix = os.path.join(self.output_dir, self.output_prefix)
        pgm_path = prefix + '.pgm'
        yaml_path = prefix + '.yaml'
        png_path = prefix + '.png'   # also save png

        w = grid.info.width
        h = grid.info.height
        res = grid.info.resolution
        ox = grid.info.origin.position.x
        oy = grid.info.origin.position.y

        # Convert OccupancyGrid data to image
        data = np.array(grid.data, dtype=np.int8).reshape((h, w))
        img = np.full((h, w), 205, dtype=np.uint8)
        img[data == 0] = 254
        img[data == 100] = 0
        img = np.flipud(img)

        # Save as PGM
        with open(pgm_path, 'wb') as f:
            f.write(f'P5\n{w} {h}\n255\n'.encode())
            f.write(img.tobytes())

        # Save as PNG using Pillow
        Image.fromarray(img).save(png_path)

        # Write YAML (still points to PGM, since ROS map_server expects PGM)
        with open(yaml_path, 'w') as f:
            f.write(f'image: {os.path.basename(pgm_path)}\n')
            f.write(f'resolution: {res}\n')
            f.write(f'origin: [{ox}, {oy}, 0.0]\n')
            f.write('negate: 0\n')
            f.write('occupied_thresh: 0.65\n')
            f.write('free_thresh: 0.196\n')

        self.get_logger().info(
            f'Map saved: {pgm_path} and {png_path} ({w}x{h}, {res}m/px, {self._map_count} updates)')


def main(args=None):
    rclpy.init(args=args)
    node = AutoMapSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Final save on shutdown
        if node._last_map is not None:
            node._save_map(node._last_map)
            node.get_logger().info('Final map saved on shutdown.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()