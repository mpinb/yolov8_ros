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


from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():

    #
    # ARGS
    #
    model = LaunchConfiguration("model")
    model_cmd = DeclareLaunchArgument(
        "model",
        default_value="/home/nci_la/soma/ros_ws/src/fly-handler/data/yolov8/segmentation_multiple_forcep_instance_all_views/runs/default_x/weights/best.pt",
        description="Model name or path")

    tracker = LaunchConfiguration("tracker")
    tracker_cmd = DeclareLaunchArgument(
        "tracker",
        default_value="bytetrack.yaml",
        description="Tracker name or path")

    device = LaunchConfiguration("device")
    device_cmd = DeclareLaunchArgument(
        "device",
        default_value="cuda:0",
        description="Device to use (GPU/CPU)")

    enable = LaunchConfiguration("enable")
    enable_cmd = DeclareLaunchArgument(
        "enable",
        default_value="True",
        description="Whether to start YOLOv8 enabled")

    threshold = LaunchConfiguration("threshold")
    threshold_cmd = DeclareLaunchArgument(
        "threshold",
        default_value="0.3",
        description="Minimum probability of a detection to be published")

    input_image_topic_left = LaunchConfiguration("input_image_topic_left")
    input_image_topic_left_cmd = DeclareLaunchArgument(
        "input_image_topic_left",
        default_value="/mid_cam/pylon_camera_node/image_raw",
        description="Name of the input image topic of left camera")

    input_image_topic_right = LaunchConfiguration("input_image_topic_right")
    input_image_topic_right_cmd = DeclareLaunchArgument(
        "input_image_topic_right",
        default_value="/right_cam/pylon_camera_node/image_raw",
        description="Name of the input image topic of right camera")    
    

    namespace = LaunchConfiguration("namespace")
    namespace_cmd = DeclareLaunchArgument(
        "namespace",
        default_value="yolo",
        description="Namespace for the nodes")

    #
    # NODES
    #
    detector_node_left_cam_cmd = Node(
        package="yolov8_ros",
        executable="yolov8_node",
        name="yolov8_node",
        namespace=f"yolo/left_cam",
        parameters=[{"model": model,
                     "device": device,
                     "enable": enable,
                     "threshold": threshold}],
        remappings=[("image_raw", input_image_topic_left)]
    )

    detector_node_right_cam_cmd = Node(
        package="yolov8_ros",
        executable="yolov8_node",
        name="yolov8_node",
        namespace=f"yolo/right_cam",
        parameters=[{"model": model,
                     "device": device,
                     "enable": enable,
                     "threshold": threshold}],
        remappings=[("image_raw", input_image_topic_right)]
    )

    tracking_node_left_cam_cmd = Node(
        package="yolov8_ros",
        executable="tracking_node",
        name="tracking_node",
        namespace=f"yolo/left_cam",
        parameters=[{"tracker": tracker}],
        remappings=[("image_raw", input_image_topic_left)]
    )

    tracking_node_right_cam_cmd = Node(
        package="yolov8_ros",
        executable="tracking_node",
        name="tracking_node",
        namespace=f"yolo/right_cam",
        parameters=[{"tracker": tracker}],
        remappings=[("image_raw", input_image_topic_right)]
    )

    debug_node_left_cam_cmd = Node(
        package="yolov8_ros",
        executable="debug_node",
        name="debug_node",
        namespace=f"yolo/left_cam",
        remappings=[("image_raw", input_image_topic_left),
                    ("detections", "tracking")]
    )

    debug_node_right_cam_cmd = Node(
        package="yolov8_ros",
        executable="debug_node",
        name="debug_node",
        namespace=f"yolo/right_cam",
        remappings=[("image_raw", input_image_topic_right),
                    ("detections", "tracking")]
    )

    ld = LaunchDescription()

    ld.add_action(model_cmd)
    ld.add_action(tracker_cmd)
    ld.add_action(device_cmd)
    ld.add_action(enable_cmd)
    ld.add_action(threshold_cmd)
    ld.add_action(input_image_topic_left_cmd)
    ld.add_action(input_image_topic_right_cmd)
    ld.add_action(namespace_cmd)

    ld.add_action(detector_node_left_cam_cmd)
    ld.add_action(detector_node_right_cam_cmd)
    ld.add_action(tracking_node_left_cam_cmd)
    ld.add_action(tracking_node_right_cam_cmd)
    ld.add_action(debug_node_left_cam_cmd)
    ld.add_action(debug_node_right_cam_cmd)

    return ld
