import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('smart_vlm'), 'config', 'params.yaml')
    return LaunchDescription([
        Node(
            package='smart_vlm',
            executable='smart_vlm_node',
            name='smart_vlm',
            output='screen',
            parameters=[{'config_path': params}],
        )
    ])
