from setuptools import setup
import os
from glob import glob

package_name = 'disaster_sensors'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Newton Adhikari',
    maintainer_email='newton@example.com',
    description='Sensor processing nodes for disaster robot simulator',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_processor  = disaster_sensors.lidar_processor:main',
            'camera_processor = disaster_sensors.camera_processor:main',
            'ekf_monitor      = disaster_sensors.ekf_monitor:main',
            'collect_ekf_data = disaster_sensors.collect_ekf_data:main',
            'rl_navigator     = disaster_sensors.rl_navigator:main',
            'frontier_explorer = disaster_sensors.frontier_explorer:main',
            'potential_field_navigator = disaster_sensors.potential_field_navigator:main',
            'benchmark_metrics = disaster_sensors.benchmark_metrics:main',
            'measure_coverage = disaster_sensors.measure_coverage:main',
            'auto_map_saver  = disaster_sensors.auto_map_saver:main',
        ],
    },
)
