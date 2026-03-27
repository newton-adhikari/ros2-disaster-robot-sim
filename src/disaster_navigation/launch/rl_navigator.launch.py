from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='explore',
                              description="'explore' (autonomous) or 'navigate' (goal-directed)"),
        DeclareLaunchArgument('goal_x',     default_value='8.0',
                              description='Navigation goal X (metres) — used in navigate mode'),
        DeclareLaunchArgument('goal_y',     default_value='12.0',
                              description='Navigation goal Y (metres) — used in navigate mode'),
        DeclareLaunchArgument('model_path', default_value='',
                              description='Path to trained PPO .pt file (leave empty for reactive policy)'),

        Node(
            package='disaster_sensors',
            executable='rl_navigator',
            name='rl_navigator',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'mode':         LaunchConfiguration('mode'),
                'goal_x':       LaunchConfiguration('goal_x'),
                'goal_y':       LaunchConfiguration('goal_y'),
                'model_path':   LaunchConfiguration('model_path'),
            }]
        ),
    ])
