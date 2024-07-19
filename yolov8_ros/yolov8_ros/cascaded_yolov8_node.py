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


from typing import List, Dict
import numpy as np

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node

from cv_bridge import CvBridge
import cv2 as cv
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks
from ultralytics.engine.results import Keypoints

from sensor_msgs.msg import Image
from yolov8_msgs.msg import Point2D
from yolov8_msgs.msg import BoundingBox2D
from yolov8_msgs.msg import Mask
from yolov8_msgs.msg import KeyPoint2D
from yolov8_msgs.msg import KeyPoint2DArray
from yolov8_msgs.msg import Detection
from yolov8_msgs.msg import DetectionArray
from std_srvs.srv import SetBool

import message_filters


class CascadedYolov8Node(Node):
    def __init__(self) -> None:
        super().__init__("cascaded_yolov8_node")

        self.device = "0"
        self.threshold = 0.1
        self.enable = True
        self.iou = 0.5

        self.cv_bridge = CvBridge()

        # pubs
        self._pub_left = self.create_publisher(
            DetectionArray, "left_cam/detections", 10
        )
        self._pub_mid = self.create_publisher(DetectionArray, "mid_cam/detections", 10)
        self._pub_right = self.create_publisher(
            DetectionArray, "right_cam/detections", 10
        )

        # Syncronize the three cameras
        self._sub_left = message_filters.Subscriber(self, Image, "left_cam/image_raw")
        self._sub_mid = message_filters.Subscriber(self, Image, "mid_cam/image_raw")
        self._sub_right = message_filters.Subscriber(self, Image, "right_cam/image_raw")

        ts = message_filters.ApproximateTimeSynchronizer(
            [self._sub_left, self._sub_mid, self._sub_right], 10, 0.1
        )

        ts.registerCallback(self.sync_image_cb)

        # services
        self._srv = self.create_service(SetBool, "enable", self.enable_cb)

        self.detection_model_path = "/home/nci_la/soma/ros_ws/src/fly-handler/data/yolov8/detect_fly_real_blue_bg/runs/exp_1_pretrainedonsynthetic_yolov8m/weights/best.pt"
        self.yolo_detect = YOLO(self.detection_model_path)

        self.pose_model_path = "/home/nci_la/soma/ros_ws/src/fly-handler/data/yolov8/pose_flyROI_real_blue_bg/runs/exp_2_pretrainedonsynthetic_yolov8m-pose/weights/best.pt"
        self.yolo_pose = YOLO(self.pose_model_path)

        # debug
        self._print_params()

    def _print_params(self) -> None:
        self.get_logger().info("detection model: {}".format(self.detection_model_path))
        self.get_logger().info("pose model: {}".format(self.pose_model_path))
        self.get_logger().info("device: {}".format(self.device))
        self.get_logger().info("threshold: {}".format(self.threshold))
        self.get_logger().info("enable: {}".format(self.enable))

    def enable_cb(
        self, req: SetBool.Request, res: SetBool.Response
    ) -> SetBool.Response:
        self.enable = req.data
        res.success = True
        return res

    def parse_hypothesis(self, results: Results) -> List[Dict]:
        hypothesis_list = []

        box_data: Boxes
        for box_data in results.boxes:
            hypothesis = {
                "class_id": int(box_data.cls),
                "class_name": self.yolo_pose.names[int(box_data.cls)],
                "score": float(box_data.conf),
            }
            hypothesis_list.append(hypothesis)

        return hypothesis_list

    def parse_boxes(self, results: Results, roi_coords) -> List[BoundingBox2D]:
        boxes_list = []

        box_data: Boxes
        for box_data in results.boxes:
            msg = BoundingBox2D()

            # get boxes values
            box = box_data.xywh[0]
            msg.center.position.x = float(box[0]) + roi_coords[0]
            msg.center.position.y = float(box[1]) + roi_coords[1]
            msg.size.x = float(box[2]) + (roi_coords[2] - roi_coords[0])
            msg.size.y = float(box[3]) + (roi_coords[3] - roi_coords[1])

            # append msg
            boxes_list.append(msg)

        return boxes_list

    def parse_keypoints(self, results: Results, roi_coords) -> List[KeyPoint2DArray]:
        keypoints_list = []

        points: Keypoints

        for points in results.keypoints:
            msg_array = KeyPoint2DArray()

            if points.conf is None:
                continue

            for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):
                if conf >= self.threshold:
                    msg = KeyPoint2D()

                    msg.id = kp_id + 1

                    msg.point.x = float(p[0]) + roi_coords[0]
                    msg.point.y = float(p[1]) + roi_coords[1]
                    msg.score = float(conf)

                    msg_array.data.append(msg)

            keypoints_list.append(msg_array)

        return keypoints_list

    def extract_roi(self, image: Image, box: BoundingBox2D) -> Image:
        cx, cy, w, h = box.xywh.cpu().numpy()[0]
        new_size = 640

        x1_new = int(max(cx - new_size / 2, 0))
        y1_new = int(max(cy - new_size / 2, 0))
        x2_new = int(min(cx + new_size / 2, image.shape[1]))
        y2_new = int(min(cy + new_size / 2, image.shape[0]))

        return image[y1_new:y2_new, x1_new:x2_new], (x1_new, y1_new, x2_new, y2_new)

    def sync_image_cb(self, left_msg: Image, mid_msg: Image, right_msg: Image) -> None:

        cv_image_left = self.cv_bridge.imgmsg_to_cv2(left_msg)
        cv_image_mid = self.cv_bridge.imgmsg_to_cv2(mid_msg)
        cv_image_right = self.cv_bridge.imgmsg_to_cv2(right_msg)

        cv_image_left = cv.cvtColor(cv_image_left, cv.COLOR_BGR2RGB)
        cv_image_mid = cv.cvtColor(cv_image_mid, cv.COLOR_BGR2RGB)
        cv_image_right = cv.cvtColor(cv_image_right, cv.COLOR_BGR2RGB)

        detection_results = self.yolo_detect.predict(
            [cv_image_left, cv_image_mid, cv_image_right],
            task="detect",
            verbose=True,
            stream=False,
            conf=0.5,
            device=self.device,
            iou=0.5,
        )

        if detection_results[0]:
            roi_left, roi_left_coords = self.extract_roi(
                cv_image_left, detection_results[0].boxes[0]
            )

        else:
            roi_left = np.zeros((640, 640, 3), np.uint8)
            roi_left_coords = (0, 0, 640, 640)

        if detection_results[1]:
            roi_mid, roi_mid_coords = self.extract_roi(
                cv_image_mid, detection_results[1].boxes[0]
            )

        else:
            roi_mid = np.zeros((640, 640, 3), np.uint8)
            roi_mid_coords = (0, 0, 640, 640)

        if detection_results[2]:
            roi_right, roi_right_coords = self.extract_roi(
                cv_image_right, detection_results[2].boxes[0]
            )
        else:
            roi_right = np.zeros((640, 640, 3), np.uint8)
            roi_right_coords = (0, 0, 640, 640)

        # cv.imshow("left", roi_left)
        # cv.imshow("mid", roi_mid)
        # cv.imshow("right", roi_right)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        pose_results = self.yolo_pose.predict(
            [roi_left, roi_mid, roi_right],
            task="pose",
            verbose=True,
            stream=False,
            conf=0.1,
            device=self.device,
            iou=0.5,
        )

        self.process_detections(
            pose_results[0], left_msg, roi_left_coords, self._pub_left
        )
        self.process_detections(pose_results[1], mid_msg, roi_mid_coords, self._pub_mid)

        self.process_detections(
            pose_results[2], right_msg, roi_right_coords, self._pub_right
        )
        self.get_logger().info("Processed detections")

    def process_detections(self, results, msg, roi_coords, pub) -> None:
        if results.boxes:
            hypothesis = self.parse_hypothesis(results)
            boxes = self.parse_boxes(results, roi_coords)

        if results.keypoints:
            keypoints = self.parse_keypoints(results, roi_coords)

        detections_msg = DetectionArray()

        for i in range(len(results)):
            aux_msg = Detection()

            if results.boxes:
                aux_msg.class_id = hypothesis[i]["class_id"]
                aux_msg.class_name = hypothesis[i]["class_name"]
                aux_msg.score = hypothesis[i]["score"]

                aux_msg.bbox = boxes[i]

            if results.keypoints:
                aux_msg.keypoints = keypoints[i]

            detections_msg.detections.append(aux_msg)

        # publish detections
        detections_msg.header = msg.header
        pub.publish(detections_msg)


def main():
    rclpy.init()
    node = CascadedYolov8Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
