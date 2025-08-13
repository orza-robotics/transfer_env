import launch
from launch.substitutions import Command, LaunchConfiguration
import launch_ros
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():


    publisher_node_planner = launch_ros.actions.Node(
        package='video_reader',
        executable='video_reader_node',
        name='video_reader_node',
        output='screen',
        additional_env={'RCUTILS_CONSOLE_OUTPUT_FORMAT': "{message}"}
    )
    
    return launch.LaunchDescription([
        publisher_node_planner
    ])