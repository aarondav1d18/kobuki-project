# from vroomba.path_planning import main as path_main
# from vroomba.control import main as control_main
# from vroomba.path_planning import *
# from vroomba.control import *

# Im just gonna really hope this works

# ROS2 includes
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber
from ament_index_python.packages import get_package_share_directory

# General imports
import os
from math import sqrt

# Various OpenCV and numpy imports required to process images
import cv2
import numpy as np
from cv_bridge import CvBridge

# Messages and services required to communicate with the camera and the Kobuki
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from .data_structs.CurrentPosition import CurrentPosition # this shit dont work outside of it

from .data_structs.ConeMap import ConeMap

# Standard ros2 messages
from std_msgs.msg import Float64MultiArray, Header

# Variables for controlling how strict we are with YOLO detections
YOLO_CONF_THESHOLD = 0.5
NMS_THRESHOLD = 0.5

DISTANCE_THRESHOLD_DUPLICATE_CONE = 0.2

MIN_DEPTH_POINTS_IN_BOX = 2

def print_red(text): print("\033[91m {}\033[00m" .format(text))
def print_yellow(text): print("\033[93m {}\033[00m".format(text))
def print_green(text): print("\033[92m {}\033[00m" .format(text))


class YOLOv4:
    """
    A class for instantiating and inferencing a YOLOv4 model, provided so you don't have to worry about messing about with the OpenCV DNN framework
    Feel free to use it, or use something else if you would prefer :) 
    """
    def __init__(self):
        # This is a little bit counter-intuitive, this allows us to dynamically load the models instead of hardcoding paths
        models_dir = os.path.join(get_package_share_directory('vroomba'), 'models')
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
        :return: An [m x n x 6] list of m frames with n cone detections each (n can vary) where each 6 element detection vector is a bounding box of the format
                [x, y, w, h, class_id, confidence]
        """ 
        # YOLO inference
        try:
            blob: np.array = cv2.dnn.blobFromImages(frames, 1.0, (416, 416), (0,0,0), swapRB=True, crop=False)
        except Exception as e:
            print(f"Error: {e}")
            return [[]]

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
    

class MapCalculator(Node):
    
    def __init__(self) -> None:
        super().__init__("map_calculator")
        self.get_logger().info('Perception node started')

        self.node_clock: Clock = Clock()
        
        self.camera_info_sub = self.create_subscription(CameraInfo, '/zed2/zed_node/right/camera_info', self.camera_info_callback, qos_profile=1)
        # Create a subscriber to the left camera frame topic
        self.zed_left_image_sub = Subscriber(self, Image, '/zed2/zed_node/left/image_rect_color')
        self.zed_depth_sub = Subscriber(self, Image, '/zed2/zed_node/depth/depth_registered')
        # TODO: determine queue size
        # note depth timestamp corresponds to the timestamp of an image it used
        # self.zed_left_image_depth_map_synch = TimeSynchronizer([self.zed_left_image_sub, self.zed_depth_sub], queue_size=20)
        self.zed_left_image_depth_map_synch = ApproximateTimeSynchronizer(
            [self.zed_left_image_sub, self.zed_depth_sub],
            queue_size=20,
            slop=0.1  # Tolerance in seconds
        )
        self.zed_left_image_depth_map_synch.registerCallback(self.seen_cones_callback)
        
        self.seen_cones_pub = self.create_publisher(Float64MultiArray, '/seen_cones', qos_profile=1)
        # WHAT THE FUCK EVEN IS THISSSSSS
        # I dont even know if these would work
        # self.seen_cones_sub = Subscriber(self, Float64MultiArray, '/seen_cones')
        
        # self.vroomba_position_sub = Subscriber(self, CurrentPosition, '/absolute_pos')
        
        # self.seen_cones_odom_change_synch = ApproximateTimeSynchronizer([self.seen_cones_sub, self.vroomba_position_sub], queue_size=20, slop=0.1)
        # self.seen_cones_odom_change_synch.registerCallback(self.update_cone_map_callback)
        
        self.cone_map = []
        # self.cone_map_pub = self.create_publisher(ConeMap, 'cone_map', qos_profile=1)
        
        # Initalise a CV Bridge - this will enable us to turn ros2 Image messages into numpy arrays 
        self.bridge = CvBridge()

        # Initialise YOLO and perform a dry run to prepare the network
        self.yolo_model = YOLOv4()

        # initialise frame
        self.last_left_frame = None
        
        # initialise projection matrix
        self.projection_matrix = None
        
        self.intrinsic_projection_matrix = None
        self.distortion_coefficients = None

        # initialise cone info message
        self.cone_pos_msg = Float64MultiArray()

        # why not just use original class id in colour map (and maybe map to names)?
        # set class map
        # maps class id that we receive to colours ids
        self.class_map = {
            0: 1, # blue is 1
            3: 2, # yellow is 2
            2: 0, # orange is 0
            1: 3, # big orange is 3
        }

        self.colour_map = {
            1: (255,0,0),
            2: (153,255,255),
            3: (0,165,255),
            0: (0,165,255),
        }


    def camera_info_callback(self, msg: CameraInfo) -> None:
        # print("Projection callback called!")
        if self.projection_matrix is None:
            self.projection_matrix = msg.p
        if self.intrinsic_projection_matrix is None:
            k = msg.k
            # https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
            self.intrinsic_projection_matrix = np.array([[k[0], k[1], k[2]],
                                                         [k[3], k[4], k[5]],
                                                         [k[6], k[7], k[8]]], dtype=np.float64)
        if self.distortion_coefficients is None:
            d = msg.d
            self.distortion_coefficients = np.array([d[0], d[1], d[2], d[3], d[4]])  # [k1, k2, t1, t2, k3]
        
        # debug
        # print(f"Projection matrix: {self.projection_matrix}")
        # for elt in self.projection_matrix:
        #     print(elt)
    
    
    def undistort_point(self, x, y):
        '''
        assumes self.intrinsic_projection_matrix and self.distortion_coefficients is not None
        '''
        src = np.array([[[x, y]]], dtype=np.float32)
        dst = cv2.undistortPoints(src, self.intrinsic_projection_matrix, self.distortion_coefficients, None, self.intrinsic_projection_matrix)
        
        return dst[0, 0]
    

    def seen_cones_callback(self, left_img_msg, depth_map_msg) -> None:
        '''
        Called when zed left image and corresponding depth map received

        Process
            - Find depth at location of bounding boxes
            - Calculate relative position of cones

        Send from publisher
        '''
        
        left_frame = self.bridge.imgmsg_to_cv2(left_img_msg, desired_encoding='bgr8')
        # for debugging
        # cv2LeftFrame = left_frame.copy()

        depth_map = self.bridge.imgmsg_to_cv2(depth_map_msg, desired_encoding='32FC1')

        bounding_boxes = self.yolo_model.predict([left_frame])[0]
        # If we don't have any bounding boxes (i.e. cones) we don't want to take any action
        if len(bounding_boxes) == 0:
            # return
            pass
        
        print("-"*10)
        map_data = []
        for box in bounding_boxes:
            image_box_x, image_box_y, image_box_w, image_box_h, class_id, confidence = box
            class_id = self.class_map[class_id]

            # define pixel coords of the centre of bounding box
            image_x, image_y = int(image_box_x + 0.5 * image_box_w), int(image_box_y + 0.5 * image_box_h)

            # draw bounding box on top of the image for debugging 
            # cv2LeftFrame = cv2.rectangle(cv2LeftFrame, (round(image_box_x), round(image_box_y)), (round(image_box_x+image_box_w), round(image_box_y+image_box_h)), self.colour_map[class_id], 2)
            
            # get depth at and around that pixel
            # depth = depth_map[image_y, image_x]
            depth_box_half_size_x = 8
            depth_box_half_size_y = 8
            if (image_x - depth_box_half_size_x < 0 or image_x + depth_box_half_size_x >= depth_map.shape[1]
            or image_y - depth_box_half_size_y < 0 or image_y + depth_box_half_size_y >= depth_map.shape[0]):
                print_yellow("Depth box out of bounds")
                continue
            # cv2LeftFrame = cv2.rectangle(cv2LeftFrame, (round(image_x) - depth_box_half_size_x, round(image_y) - depth_box_half_size_y), (round(image_x+depth_box_half_size_x), round(image_y+depth_box_half_size_y)), (100, 100, 100), 2)

            # depth_box = depth_map[image_y-depth_box_half_size_y:image_y+depth_box_half_size_y, image_x-depth_box_half_size_x:image_x+depth_box_half_size_x]
            # depth_box = np.nan_to_num(depth_box, nan=0.0, posinf=0.0, neginf=0.0)
            # non_zero_depth_box_count = np.count_nonzero(depth_box)
            # if non_zero_depth_box_count < MIN_DEPTH_POINTS_IN_BOX:
            #     print_yellow("Depth box has too few non zero (and non NaN) values.")
            #     continue
            # depth = np.sum(depth_box) / non_zero_depth_box_count

            depth = depth_map[image_y, image_x]
            
            # # get relevant values of projection matrix
            # p = self.projection_matrix
            # if p is not None:    
            #     fx = p[0]
            #     cx = p[2]
            #     fy = p[5]
            #     cy = p[6]
            # else:
            #     break

            # # find real world coords
            # xw = (image_x - cx) * z / fx
            # y_dash = (image_y - cy) * z / fy
            # zw = z

            # try:
            #     yw = sqrt(z**2 - y_dash**2 - xw**2)
            # except ValueError as e:
            #     print(f"Value error: {e}")
            #     print(xw, y_dash, zw)
            #     yw = 0
            
            # yw = z
            # xw = (image_x - cx) * yw / fx
            # y_dash, zw = 0, 0
            
            if self.intrinsic_projection_matrix is None or self.distortion_coefficients is None:
                print("No intrinsic projection matrix or distortion coefficients")
                break
            
            # also x in top down view
            world_x = (image_x - self.intrinsic_projection_matrix[0, 2]) * depth / self.intrinsic_projection_matrix[0, 0]
            # do not confuse this with the y in top down view; basically height
            world_y = (image_y - self.intrinsic_projection_matrix[1, 2]) * depth / self.intrinsic_projection_matrix[1, 1]
            
            mapped_x = world_x
            mapped_y = depth**2 - world_y**2 - world_x**2
            if mapped_y < 0:
                print_red("Mapped_y to a negative value (cone would be behind the zed2 camera)")
            else:
                mapped_y = sqrt(mapped_y)
            # append a cone data to the map
            map_data += [np.float64(mapped_x), np.float64(mapped_y), np.float64(class_id)]
            
            # debugging data
            # print(fx, cx, fy, cy, z, u)
            # print(f"Cone at {x + 0.5 * w, y + 0.5 * h}:")
            # print("Original image vals")
            # print(image_x, image_y)
            # print("Prev real world final data")
            # print(xw, yw, zw)
            # print("Real world coords:")
            # print(world_x, world_y, depth, depth**2 - world_y**2)  # last value should be sqrted
            # print("Mapped values (x, y):")
            # print(mapped_x, mapped_y)
            # print("Class and confidence:")
            # print(f"{class_id} ({'blue' if class_id == 1 else 'yellow' if class_id == 2 else 'orange' if class_id == 0 else 'big orange'}), {confidence}, \n")

            # mark centre of bounding box if we have depth at that point
            # cv2LeftFrame = cv2.circle(cv2LeftFrame, (round(image_x), round(image_y)), 5, (0, 255, 0), 3)

        # publish x, y and class_id values
        # cone_pos_msg = Float64MultiArray()
        self.cone_pos_msg.data = map_data
        self.seen_cones_pub.publish(self.cone_pos_msg)

        # show an image for debugging
        # print(f"Left image timestamp: {left_img_msg.header.stamp}")
        # print(f"Depth map timestamp: {depth_map_msg.header.stamp}")
        # cv2.imshow("image", cv2LeftFrame)
        # cv2.imshow("depth", depth_map)
        # cv2.waitKey(0)
        print("-"*10)
        
        
    def update_cone_map_callback(self, seen_cones_msg, prev_pos_msg) -> None:
        '''
        Called when seen_cones and pos_change_accum received

        Process
            - Update cone map with new cone positions
            - Publish new cone map
        '''
        # print("Update cone map callback called!")
        # print(f"Seen cone pos: {seen_cones_msg}")
        # print(f"Pos change accum: {pos_change_accum_msg}")
        new_header = Header()
        new_header.stamp = self.node_clock.now().to_msg()
        
        prev_pos = np.array([prev_pos_msg.posx, prev_pos_msg.posy])  # position at the time (approximately) of then the cones were mapped to a frame
        new_cone_pos = np.reshape(np.array(seen_cones_msg.data), (-1, 4))
        # map to the absolute coordinate system
        new_cone_pos[:, 0:2] = np.matmul(np.array([[np.cos(prev_pos_msg.yaw), np.sin(prev_pos_msg.yaw)], [-np.sin(prev_pos_msg.yaw), np.cos(prev_pos_msg.yaw)]]), new_cone_pos[:, 0:2].T).T
        new_cone_pos[:, 0:2] += prev_pos
        # TODO: make better performance wise
        for cone in new_cone_pos:
            in_map = False
            for cone_in_map in self.cone_map:
                difference_in_distance = np.linalg.norm(cone[0:2] - cone_in_map[0:2])
                if difference_in_distance <= DISTANCE_THRESHOLD_DUPLICATE_CONE:
                    in_map = True
                    break
            
            if not in_map:
                self.cone_map += new_cone_pos.tolist()
        
        cone_map_msg = ConeMap()
        cone_map_msg.header = new_header
        cone_array = Float64MultiArray()
        cone_array.data = self.cone_map
        cone_map_msg.cones = cone_array
        
        self.cone_map_pub.publish(cone_map_msg)


def main(args=None) -> None:
    """
    The entrypoint of the program, this will create the ROS2 node, tell the ROS2 runtime to 'spin' it
    and then shutdown gracefully when the node quits
    """
    rclpy.init(args=args)
    vroomba_node = MapCalculator()
    rclpy.spin(vroomba_node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    # path_main()
    # control_main()