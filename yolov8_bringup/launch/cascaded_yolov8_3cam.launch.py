# Copyright (C) 2023  Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    model = LaunchConfiguration("model")
    model_cmd = DeclareLaunchArgument(
        "model",
        default_value="/home/nci_la/soma/ros_ws/src/fly-handler/data/yolov8/pose_flyROI_real_blue_bg/runs/exp_2_pretrainedonsynthetic_yolov8m-pose/weights/best.pt",
        description="Model name or path",
    )

    tracker = LaunchConfiguration("tracker")
    tracker_cmd = DeclareLaunchArgument(
        "tracker", default_value="bytetrack.yaml", description="Tracker name or path"
    )

    device_1 = LaunchConfiguration("device_1")
    device_1_cmd = DeclareLaunchArgument(
        "device_1",
        default_value="cuda:0",
        description="Device to use (GPU/CPU)",
    )

    device_2 = LaunchConfiguration("device_2")
    device_2_cmd = DeclareLaunchArgument(
        "device_2",
        default_value="cuda:1",
        description="Device to use (GPU/CPU)",
    )

    device_3 = LaunchConfiguration("device_3")
    device_3_cmd = DeclareLaunchArgument(
        "device_3",
        default_value="cuda:2",
        description="Device to use (GPU/CPU)",
    )

    enable = LaunchConfiguration("enable")
    enable_cmd = DeclareLaunchArgument(
        "enable", default_value="True", description="Whether to start YOLOv8 enabled"
    )

    threshold = LaunchConfiguration("threshold")
    threshold_cmd = DeclareLaunchArgument(
        "threshold",
        default_value="0.1",
        description="Minimum probability of a detection to be published",
    )

    filter_freq = LaunchConfiguration("filter_freq")
    filter_freq_cmd = DeclareLaunchArgument(
        "filter_freq", default_value="20", description="Filter frequency"
    )
    filter_mincutoff = LaunchConfiguration("filter_mincutoff")
    filter_mincutoff_cmd = DeclareLaunchArgument(
        "filter_mincutoff", default_value="0.5", description="Filter mincutoff"
    )

    filter_beta = LaunchConfiguration("filter_beta")
    filter_beta_cmd = DeclareLaunchArgument(
        "filter_beta", default_value="0.0", description="Filter beta"
    )

    filter_dcutoff = LaunchConfiguration("filter_dcutoff")
    filter_dcutoff_cmd = DeclareLaunchArgument(
        "filter_dcutoff", default_value="1.0", description="Filter dcutoff"
    )

    input_image_topic_left = LaunchConfiguration("input_image_topic_left")
    input_image_topic_left_cmd = DeclareLaunchArgument(
        "input_image_topic_left",
        default_value="/left_cam/pylon_camera_node/image_raw",
        description="Name of the input image topic of left camera",
    )

    input_image_topic_right = LaunchConfiguration("input_image_topic_right")
    input_image_topic_right_cmd = DeclareLaunchArgument(
        "input_image_topic_right",
        default_value="/right_cam/pylon_camera_node/image_raw",
        description="Name of the input image topic of right camera",
    )

    input_image_topic_mid = LaunchConfiguration("input_image_topic_mid")
    input_image_topic_mid_cmd = DeclareLaunchArgument(
        "input_image_topic_mid",
        default_value="/mid_cam/pylon_camera_node/image_raw",
        description="Name of the input image topic of mid camera",
    )

    namespace = LaunchConfiguration("namespace")
    namespace_cmd = DeclareLaunchArgument(
        "namespace",
        default_value="yolo",
        description="Namespace for the nodes",
    )

    # Nodes
    cascaded_yolo_node_combined_cmd = Node(
        package="yolov8_ros",
        executable="cascaded_yolov8_node",
        name="cascaded_yolov8_node",
        namespace=namespace,
        parameters=[
            {
                "model": model,
                "device": device_1,
                "enable": enable,
                "threshold": threshold,
            }
        ],
        remappings=[
            ("left_cam/image_raw", input_image_topic_left),
            ("mid_cam/image_raw", input_image_topic_mid),
            ("right_cam/image_raw", input_image_topic_right),
        ],
    )

    tracking_node_left_cam_cmd = Node(
        package="yolov8_ros",
        executable="tracking_node",
        name="tracking_node",
        namespace=f"yolo/left_cam",
        parameters=[
            {
                "tracker": tracker,
                "filter_freq": filter_freq,
                "filter_mincutoff": filter_mincutoff,
                "filter_beta": filter_beta,
                "filter_dcutoff": filter_dcutoff,
            }
        ],
        remappings=[
            ("image_raw", input_image_topic_left),
        ],
    )

    tracking_node_right_cam_cmd = Node(
        package="yolov8_ros",
        executable="tracking_node",
        name="tracking_node",
        namespace=f"yolo/right_cam",
        parameters=[
            {
                "tracker": tracker,
                "filter_freq": filter_freq,
                "filter_mincutoff": filter_mincutoff,
                "filter_beta": filter_beta,
                "filter_dcutoff": filter_dcutoff,
            }
        ],
        remappings=[
            ("image_raw", input_image_topic_right),
        ],
    )

    tracking_node_mid_cam_cmd = Node(
        package="yolov8_ros",
        executable="tracking_node",
        name="tracking_node",
        namespace=f"yolo/mid_cam",
        parameters=[
            {
                "tracker": tracker,
                "filter_freq": filter_freq,
                "filter_mincutoff": filter_mincutoff,
                "filter_beta": filter_beta,
                "filter_dcutoff": filter_dcutoff,
            }
        ],
        remappings=[
            ("image_raw", input_image_topic_mid),
        ],
    )

    debug_node_left_cam_cmd = Node(
        package="yolov8_ros",
        executable="debug_node",
        name="debug_node",
        namespace=f"yolo/left_cam",
        remappings=[("image_raw", input_image_topic_left), ("detections", "tracking")],
    )

    debug_node_right_cam_cmd = Node(
        package="yolov8_ros",
        executable="debug_node",
        name="debug_node",
        namespace=f"yolo/right_cam",
        remappings=[("image_raw", input_image_topic_right), ("detections", "tracking")],
    )

    debug_node_mid_cam_cmd = Node(
        package="yolov8_ros",
        executable="debug_node",
        name="debug_node",
        namespace=f"yolo/mid_cam",
        remappings=[("image_raw", input_image_topic_mid), ("detections", "tracking")],
    )

    ld = LaunchDescription()

    ld.add_action(model_cmd)
    ld.add_action(tracker_cmd)
    ld.add_action(device_1_cmd)
    ld.add_action(device_2_cmd)
    ld.add_action(device_3_cmd)
    ld.add_action(enable_cmd)
    ld.add_action(threshold_cmd)
    ld.add_action(input_image_topic_left_cmd)
    ld.add_action(input_image_topic_right_cmd)
    ld.add_action(input_image_topic_mid_cmd)
    ld.add_action(namespace_cmd)
    ld.add_action(filter_freq_cmd)
    ld.add_action(filter_mincutoff_cmd)
    ld.add_action(filter_beta_cmd)
    ld.add_action(filter_dcutoff_cmd)

    ld.add_action(cascaded_yolo_node_combined_cmd)
    ld.add_action(tracking_node_left_cam_cmd)
    ld.add_action(tracking_node_mid_cam_cmd)
    ld.add_action(tracking_node_right_cam_cmd)
    ld.add_action(debug_node_left_cam_cmd)
    ld.add_action(debug_node_mid_cam_cmd)
    ld.add_action(debug_node_right_cam_cmd)

    return ld
