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

        ],
    },
    # Python dependencies (install with pip install -r requirements.txt)
    # ultralytics, opencv-python-headless, numpy
)
