import launch
from  launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    zed_wrapper_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('zed_wrapper'), 'launch'),
            '/zed2.launch.py'])
        )

    return launch.LaunchDescription([
        zed_wrapper_launch,
        Node(
              package='vroomba',
              executable='control',
              name='control'),
        Node(
			package='vroomba',
			executable='path_planning',
			name='path_planning'),
		Node(
			package='vroomba',
			executable='perception',
			name='perception'),
        Node(
            package='ugrdv_kobuki_ros',
            executable='ugrdv_kobuki_ros',
            name='ugrdv_kobuki_ros'),
    ])