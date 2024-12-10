# ROS2 includes
import rclpy
from rclpy.node import Node

from ugrdv_kobuki_msgs.msg import Odometry, CurrentPosition

from std_msgs.msg import Header

def print_green(text): print("\033[92m {}\033[00m" .format(text))

class position(Node):
    
    def __init__(self):
        super().__init__('position_tracker')
        self.odometry_sub = self.create_subscription(Odometry, '/ugrdv_kobuki/odometry', self.odometry_callback, 4)
        
        self.cur_pos_pub = self.create_publisher(CurrentPosition, '/absolute_pos', 1)
        
        header = Header()
        header.stamp = self.node_clock.now().to_msg()        
        self.cur_pos = (header, 0, 0)
        self.cur_pos_pub.publish(CurrentPosition(header=self.cur_pos[0], posx=self.cur_pos[1], posy=self.cur_pos[2]))
    
    
    def odometry_callback(self, msg: Odometry) -> None:
        # print("Odometry callback called!")
        self.cur_pos = (self.msg.header, self.cur_pos[1] + msg.posx, self.cur_pos[2] + msg.posy)
        print_green(f"Odometry callback called! {self.cur_pos}")
        self.cur_pos_pub.publish(CurrentPosition(header=self.cur_pos[0], posx=self.cur_pos[1], posy=self.cur_pos[2], yaw=msg.yaw))
    