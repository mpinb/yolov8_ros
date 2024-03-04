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


class Yolov8Node(Node):
    def __init__(self) -> None:
        super().__init__("yolov8_node")

        # params
        self.declare_parameter("model", "yolov8m.pt")
        model = self.get_parameter("model").get_parameter_value().string_value

        self.declare_parameter("device", "cuda:0")
        self.device = self.get_parameter("device").get_parameter_value().string_value

        self.declare_parameter("threshold", 0.4)
        self.threshold = (
            self.get_parameter("threshold").get_parameter_value().double_value
        )

        self.declare_parameter("enable", True)
        self.enable = self.get_parameter("enable").get_parameter_value().bool_value

        self.iou = 0.7

        self.cv_bridge = CvBridge()
        self.yolo = YOLO(model)
        self.yolo.fuse()

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

        # debug
        self.__debug(model)

    def __debug(self, model) -> None:
        self.get_logger().info("model: {}".format(model))
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
                "class_name": self.yolo.names[int(box_data.cls)],
                "score": float(box_data.conf),
            }
            hypothesis_list.append(hypothesis)

        return hypothesis_list

    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:
        boxes_list = []

        box_data: Boxes
        for box_data in results.boxes:
            msg = BoundingBox2D()

            # get boxes values
            box = box_data.xywh[0]
            msg.center.position.x = float(box[0])
            msg.center.position.y = float(box[1])
            msg.size.x = float(box[2])
            msg.size.y = float(box[3])

            # append msg
            boxes_list.append(msg)

        return boxes_list

    def parse_masks(self, results: Results) -> List[Mask]:
        masks_list = []

        def create_point2d(x: float, y: float) -> Point2D:
            p = Point2D()
            p.x = x
            p.y = y
            return p

        mask: Masks
        for mask in results.masks:
            msg = Mask()

            msg.data = [
                create_point2d(float(ele[0]), float(ele[1]))
                for ele in mask.xy[0].tolist()
            ]
            msg.height = results.orig_img.shape[0]
            msg.width = results.orig_img.shape[1]

            masks_list.append(msg)

        return masks_list

    def parse_keypoints(self, results: Results) -> List[KeyPoint2DArray]:
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
                    msg.point.x = float(p[0])
                    msg.point.y = float(p[1])
                    msg.score = float(conf)

                    msg_array.data.append(msg)

            keypoints_list.append(msg_array)

        return keypoints_list

    def sync_image_cb(self, left_msg: Image, mid_msg: Image, right_msg: Image) -> None:

        cv_image_left = self.cv_bridge.imgmsg_to_cv2(left_msg)
        cv_image_mid = self.cv_bridge.imgmsg_to_cv2(mid_msg)
        cv_image_right = self.cv_bridge.imgmsg_to_cv2(right_msg)

        cv_image_left = cv.cvtColor(cv_image_left, cv.COLOR_BGR2RGB)
        cv_image_mid = cv.cvtColor(cv_image_mid, cv.COLOR_BGR2RGB)
        cv_image_right = cv.cvtColor(cv_image_right, cv.COLOR_BGR2RGB)

        results = self.yolo.predict(
            [cv_image_left, cv_image_mid, cv_image_right],
            verbose=True,
            stream=False,
            conf=self.threshold,
            device=self.device,
            iou=self.iou,
        )

        self.process_detections(results[0], left_msg, self._pub_left)
        self.process_detections(results[1], mid_msg, self._pub_mid)
        self.process_detections(results[2], right_msg, self._pub_right)

    def process_detections(self, results, msg, pub) -> None:

        if results.boxes:
            hypothesis = self.parse_hypothesis(results)
            boxes = self.parse_boxes(results)

        if results.masks:
            masks = self.parse_masks(results)

        if results.keypoints:
            keypoints = self.parse_keypoints(results)

        detections_msg = DetectionArray()

        for i in range(len(results)):
            aux_msg = Detection()

            if results.boxes:
                aux_msg.class_id = hypothesis[i]["class_id"]
                aux_msg.class_name = hypothesis[i]["class_name"]
                aux_msg.score = hypothesis[i]["score"]

                aux_msg.bbox = boxes[i]

            if results.masks:
                aux_msg.mask = masks[i]

            if results.keypoints:
                aux_msg.keypoints = keypoints[i]

            detections_msg.detections.append(aux_msg)

        # publish detections
        detections_msg.header = msg.header
        pub.publish(detections_msg)


def main():
    rclpy.init()
    node = Yolov8Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
