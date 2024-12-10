"""
This is an example ROS2 node using the UGRDV Kobuki ROS2 API. Its purpose is to demonstrate the full-stack functionality of the platform, including:
- Receiving camera images
- Processing camera images on the GPU
- Sending drive commands

It will follow the cone nearest to the centre of the camera frame.

One important thing to note is, if you do type annotations (you should try to!), avoid using `list[int] or tuple[int, int]` type annotations
 as the Koubki runs python3.6 which will crash when trying to parse these type annotations. 
You can still use type hints such as the following:
 - `def function(...) -> float`
 - etc...

"""

# ROS2 includes
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

# Various OpenCV and numpy imports required to process images
import cv2
import numpy as np
from cv_bridge import CvBridge

# Messages and services required to communicate with the camera and the Kobuki
from ugrdv_kobuki_msgs.msg import DriveCommand
from ugrdv_kobuki_msgs.srv import EnableDrive
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose, Point
from builtin_interfaces.msg import Duration
from ugrdv_kobuki_msgs.msg import Cone, ConeArray
from .path_planning import TrackNetwork, NodeType
import time

# General imports
import os

# Constants for how fast the Kobuki should rotate and move laterally
ANGULAR_VELOCITY_MULTIPLIER = 0.001
CONSTANT_LINEAR_VELOCITY = 0.0

# Variables for controlling how strict we are with YOLO detections
YOLO_CONF_THESHOLD = 0.5
NMS_THRESHOLD = 0.5


class YOLOv4:
    """
    A class for instantiating and inferencing a YOLOv4 model, provided so you don't have to worry about messing about with the OpenCV DNN framework
    Feel free to use it, or use something else if you would prefer :) 
    """
    def __init__(self):
        # This is a little bit counter-intuitive, this allows us to dynamically load the models instead of hardcoding paths
        models_dir = os.path.join(get_package_share_directory('kobuki_example'), 'models')
        self.yolo_net = cv2.dnn.readNet(os.path.join(models_dir, "yolov4-tiny-ugrdv-416_best.weights"), os.path.join(models_dir, "yolov4-tiny-ugrdv-416.cfg"))
        self.yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Dry-run to initialise network
        self.predict([np.zeros((416, 416, 3), dtype=np.uint8)])

    def predict(self, frames):
        """
        Inferences the YOLO network on batched image(s), note both the images MUST have the same dimensions

        :frames: A n long list of [h x w x 3] tensors, where n is the number of frames to detect frames in, h is the height of each frame
                w is the width of each frame and there are 3 colour channels
        :return: An [n x m x 6] list of n cone detections from m frames where each 6 element detection vector is a bounding box of the format
                [x, y, w, h, class_id, confidence]
        """ 
        # YOLO inference
        blob: np.array = cv2.dnn.blobFromImages(frames, 1.0, (416, 416), (0,0,0), swapRB=True, crop=False)

        # Set the image at the network input and scale the image to range [0, 1] from [0, 255]
        self.yolo_net.setInput(blob, "", 1/255)
        frame_predictions: tuple = self.yolo_net.forward(self.yolo_net.getUnconnectedOutLayersNames()[-1])

        # Inference with only 1 frame leads to a shape of (2048, 9) we want (1, 2048, 9) to keep our code the same
        if len(frames) == 1:
            frame_predictions = [frame_predictions]
        final_boxes = [np.array([]) for _ in range(len(frame_predictions))]

        for frame_idx, frame_pred in enumerate(frame_predictions):
            bounding_boxes: list = []
            classes: list = []
            confidences: list = []

            # Filter yolo detections based on confidence
            accepted_boxes: np.array = frame_pred[np.max(frame_pred[:, 5:], axis=1) > YOLO_CONF_THESHOLD]
            for box in accepted_boxes:
                centre_x: float = box[0] * frames[0].shape[1]
                centre_y: float = box[1] * frames[0].shape[0]
                width   : float = box[2] * frames[0].shape[1]
                height  : float = box[3] * frames[0].shape[0]
                x: float = centre_x - width / 2
                y: float = centre_y - height / 2

                classes.append(np.argmax(box[5:]))
                confidences.append(np.max(box[5:]))
                bounding_boxes.append(np.array([x, y, width, height]))

            bounding_boxes: np.array = np.array(bounding_boxes)
            classes: np.array = np.array(classes)
            confidences: np.array = np.array(confidences)

            # Perform NMS suppression to get final bounding boxes
            valid_indices: np.array = cv2.dnn.NMSBoxes(tuple(bounding_boxes), confidences, YOLO_CONF_THESHOLD, NMS_THRESHOLD)

            # Construct an array of bounding boxes, where each bounding box is comprised of the following structure:
            #   [x, y, w, h, class, conf]
            if len(valid_indices) > 0:
                n_boxes = valid_indices.shape[0]
                frame_boxes: np.array = np.zeros((n_boxes, 6))
                frame_boxes[:, 0:4] = np.reshape(bounding_boxes[valid_indices], (n_boxes, 4))
                frame_boxes[:, 4] = np.reshape(classes[valid_indices], (n_boxes))
                frame_boxes[:, 5] = np.reshape(confidences[valid_indices], (n_boxes))
                final_boxes[frame_idx] = frame_boxes

        return final_boxes


class ExampleKobukiNode(Node):
    def __init__(self) -> None:
        """
        The initialisation function of the node, this does a bunch of stuff and it should all be commented but it roughly boils down to:
        1. Instantiate subscriptions and publishers required for the function of the node
        2. Enable the Kobuki drive-train 
        """

        super().__init__('kobuki_example_node')

        # Create subscriptions to the left camera frame topic
        self.camera_subscription = self.create_subscription(Image, '/zed2/zed_node/left/image_rect_color', self.camera_frame_callback, 1)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/zed2/zed_node/left/camera_info', self.camera_info_callback, 1)
        self.drive_command_publisher = self.create_publisher(DriveCommand, '/ugrdv_kobuki/drive_command', 1)
        self.marker_pub = self.create_publisher(MarkerArray, "/kobuki_soup/cone_markers", 1)
        self.cone_buffer = []
        self.state = "seeing"
        self.state_start = time.time()
        # Initalise a CV Bridge - this will enable us to turn ros2 Image messages into numpy arrays 
        self.bridge = CvBridge()
        self.camera_info = CameraInfo()
        self.camera_info.k = np.eye(3)
        self.turning_speed = 0.1
        self.driving_speed = 0.2
        self.driving_end = None
        self.turning_end = None
        self.turning_dir = None

        # Initialise YOLO and perform a dry run to prepare the network
        self.yolo_model = YOLOv4()
        # Enable the drivetrain - this will tell the ugrdv_kobuki_ros package that we want to be able to move the car
        client = self.create_client(EnableDrive, "ugrdv_kobuki/enable_drive")
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Cannot find drive-train service, waiting')
        request = EnableDrive.Request()
        request.enable = True

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().warn('Drive-train service call failed')


    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_info = msg
        self.projection_matrix = np.reshape(msg.k, (3,3))

    def camera_frame_callback(self, msg: Image) -> None:
        """
        This function will be called when we receive a camera frame

        :param msg: The message received on the image frame topic
        """

        if self.state == "seeing":

            # Convert from a ROS message to an OpenCV object
            right_numpy = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            bounding_boxes = self.yolo_model.predict([right_numpy])[0]

            # If we don't have any bounding boxes (i.e. cones) we don't want to take any action
            if len(bounding_boxes) < 4:
                return

            cones, markers = self.estimate_position_plane(
                np.reshape(self.camera_info.k, (3,3)),
                bounding_boxes
            )
            for per in cones.cones:
                closest_dist = np.inf
                closest_cone = None
                for cones in self.cone_buffer:
                    pos = np.mean(cones, axis=0)
                    dist = np.linalg.norm(pos - [per.position.x, per.position.y])
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_cone = cones
                if closest_cone and closest_dist < 0.5:
                    closest_cone.append([per.position.x, per.position.y])
                else:
                    self.cone_buffer.append([[per.position.x, per.position.y]])

            # self.marker_pub.publish(markers)
            # points = np.array([
            # [c.position.x, c.position.y] for c in cones.cones
            # ])
            # node_types = [
            #     NodeType.BLUE_CONE for _ in cones.cones
            # ]
            means = [
                np.mean(arr, axis=0) for arr in self.cone_buffer
            ]
            mArr = MarkerArray(markers=[
                Marker(pose=Pose(position=Point(x=cone[0], y=cone[1], z=0.0))) for cone in means
            ])
            for marker in mArr.markers:
                marker.header.frame_id = "map"
                marker.id = np.random.randint(1, 10000)
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 1.0
                marker.color.a = 1.0
                marker.scale.x = 1.0
                marker.scale.y = 1.0
                marker.scale.z = 1.0
                marker.type = Marker.CUBE
                marker.lifetime = Duration(sec=5)
            self.marker_pub.publish(mArr)
            
            points = np.array(means)
            if points.shape[0] < 4: return
            node_types = [NodeType.UNKNOWN_CONE for _ in points]

          
            if time.time() - self.state_start < 10.0: return


            net = TrackNetwork(points, node_types)
            try:
                cost, path = net.beam_search(
                    3,
                    3
                )[0]
            except Exception as e:
                self.get_logger().info(f"{e}")
                return
            if len(path) < 2: return
            self.state = "turning"
            self.state_start = time.time()
            self.get_logger().info(self.state)
            path = [net.get_edge_vertex(edge) for edge in path]
            point = path[1]
            angle = np.arctan2(point[1], point[0])
            turning_time = abs(angle / self.turning_speed)
            self.turning_end = time.time() + turning_time
            self.turn_direction = np.sign(angle)
            self.drive_distance = np.linalg.norm(point)
        elif self.state == "turning":
            if time.time() >= self.turning_end:
                # transition
                self.driving_end = time.time() + self.drive_distance / self.driving_speed
                self.state = "driving"
                self.state_start = time.time()
                self.get_logger().info(self.state)
            else:
                drive_command = DriveCommand()
                drive_command.header.frame_id = ''
                drive_command.header.stamp.sec = 0
                drive_command.header.stamp.nanosec = 0
                drive_command.angularvel = self.turn_direction * self.turning_speed
                drive_command.linearvel = 0.0
                # Actually publish the message to the topic, the kobuki should now drive
                self.drive_command_publisher.publish(drive_command)
        elif self.state == "driving":
            if time.time() >= self.driving_end:
                self.state = "seeing"
                self.state_start = time.time()
                self.cone_buffer = []
                self.get_logger().info(self.state)
                drive_command = DriveCommand()
                drive_command.header.frame_id = ''
                drive_command.header.stamp.sec = 0
                drive_command.header.stamp.nanosec = 0
                drive_command.angularvel = 0.0
                drive_command.linearvel = 0.0
                self.drive_command_publisher.publish(drive_command)
            else:
                drive_command = DriveCommand()
                drive_command.header.frame_id = ''
                drive_command.header.stamp.sec = 0
                drive_command.header.stamp.nanosec = 0
                drive_command.angularvel = 0.0
                drive_command.linearvel = self.driving_speed
                
        
                # Actually publish the message to the topic, the kobuki should now drive
                self.drive_command_publisher.publish(drive_command)

    def estimate_position_plane(self, left_proj, left_yolo: list) -> MarkerArray:
        """
        If we assume that the cones lie on a plane relative to the camera, we can take the direction vector from the camera and figure out where this intersects with the plane
        to get a 3D position.
        Just as a side note, while this was very accurate at autocross in 2023, I've opted not to use it in the combined camera pipeline as it makes too many assumptions and requires
        an annoying amount of fine tuning - keeping this here though, both for future reference and in-case we need it in the future.

        :param left_proj: The (3x4) projection matrix for the rectified left image
        :param left_yolo: An array of YOLO boxes representing the left frame
        :returns: An a tuple where:
                    The first element is an array of 3D cones detected in the camera frame
                    The second element is an array parallel to the first which contains the class probabilities as detected by the YOLO model
                        this will be removed when #569 is resolved
        """
        PLANE_HEIGHT = -0.27
        CAMERA_YAW = 0.0
        cones: list[Marker] = []
        for box in left_yolo:
            # Calculate the intersection at the base of the cone
            x_pos = box[0] + (box[2] // 2)
            y_pos = box[1] + box[3]

            # Calculate the direction vector from the camera passing through this pixel
            pixel = np.array([x_pos, y_pos, 1])
            intrinsic_inv = np.linalg.inv(left_proj)
            direction = intrinsic_inv @ pixel
            direction /= np.linalg.norm(direction, ord=2)
            
            # Working in left frame space, we calculate the point of intersection of this vector (from the origin) and some plane
            # This is a bit of a hack, but rather than defining the full plane & line equations, we can just assume that the plane only has a -z displacement and rotates
            #  so we can just rotate the vector and calculate the point of intersection with a flat plane with some -z component (when some plane intersects the X-Y aligned at -z)
            #  this simplifies the code
            x_theta = np.deg2rad(0.0)
            y_theta = np.deg2rad(0.0)
            z_theta = np.deg2rad(CAMERA_YAW)
            x_rot = np.array([
                [1, 0, 0],
                [0, np.cos(x_theta), -np.sin(x_theta)],
                [0, np.sin(x_theta), np.cos(x_theta)]
            ])
            y_rot = np.array([
                [np.cos(y_theta), 0, np.sin(y_theta)],
                [0, 1, 0],
                [-np.sin(y_theta), 0, np.cos(y_theta)]
            ])
            z_rot = np.array([
                [np.cos(z_theta), -np.sin(z_theta), 0],
                [np.sin(z_theta), np.cos(z_theta), 0],
                [0, 0, 1]
            ])

            direction = (x_rot @ y_rot @ z_rot) @ direction

            # Assuming plane is axis aligned and camera height is h meters above ground (plane.y = -h), get 3d point of intersection
            # This gives us a plane equation of 0x + 1y + 0z = -h
            # Substituting the parametrised vector into the plane equation we get t = -h/y, then multiply the vector by t to get the point of intersection
            t = PLANE_HEIGHT/direction[1]
            intersection = direction * t

            # Convert from this coordinate system to ugrdv_zed's
            point = Marker()
            point.pose.position.x = intersection[0]
            point.pose.position.y = intersection[2]
            point.pose.position.z = intersection[1]
            cone: Marker = Marker(
                pose=Pose(position=Point(x=intersection[0], y=intersection[2], z=intersection[1])), 
            )
            cone.lifetime = Duration(nanosec=1000000000//5)
            cone.id = np.random.randint(0, 10000)
            cone.type = Marker.CUBE
            cone.color.r = 1.0
            cone.color.g = 1.0
            cone.color.b = 1.0
            cone.color.a = 1.0
            cone.scale.x = 1.0
            cone.scale.y = 1.0
            cone.scale.z = 1.0
            cone.header.frame_id = "map"
            cones.append(cone)

        markers = MarkerArray(markers=cones)
        cones = ConeArray(
            cones = [
                Cone(position=Point(x=m.pose.position.x, y=m.pose.position.y, z=m.pose.position.z)) for m in markers.markers
            ]
        )
        # Warning - the outputs of the plane positions estimation will be axis aligned and not in the ugrdv_zed frame, they should not be merged or published without first converting to ugrdv_zed
        return cones, markers 


def main(args=None) -> None:
    """
    The entrypoint of the program, this will create the ROS2 node, tell the ROS2 runtime to 'spin' it
    and then shutdown gracefully when the node quits
    """
    rclpy.init(args=args)
    kobuki_example_node = ExampleKobukiNode()
    rclpy.spin(kobuki_example_node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
