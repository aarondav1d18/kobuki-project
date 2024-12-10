import rclpy
from rclpy.node import Node

##imports for the kobuki information
from ugrdv_kobuki_msgs.msg  import Odometry, DriveCommand
from ugrdv_kobuki_msgs.srv import EnableDrive, SetLED, PlaySound
from std_msgs.msg import Float64MultiArray
import numpy as np
from .data_structs.Pos import Point

ORANGE_SEEN_AT_FINISH = 1
MAX_LINEAR = 1.0
MAX_ANGULAR = np.pi
class Control(Node):
    def __init__(self)-> None:
            super().__init__('Control')
            self.get_logger().info('Control node started')


            self.num_times_seen_orange = 0
            self.prev_cone_type_seen = 0

            self.drive_command_publisher = self.create_publisher(
                 DriveCommand,
                '/ugrdv_kobuki/drive_command', 
                1
            )

            #Create a client variable for the LED
            self.target_point_pos = self.create_subscription(
                 Float64MultiArray,
                 '/velocity_info',
                 self.compute_control,
                 1
            )

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
            self.get_logger().info('AHHHHHHH WE ARE MOVING')

    def compute_control(self, msg: Float64MultiArray) -> None:
        """
        Compute linear and angular velocities to move the car to the target point.

        Parameters:
        - current_pose: tuple (x_c, y_c, theta_c), the current position and orientation of the car.
        - target_point: tuple (x_t, y_t), the target point in 2D space.

        Returns:
        - v: float, linear velocity.
        - omega: float, angular velocity.
        """
        msg = msg.data
        k_p=1.0 # Proportional gain for linear velocity. May want to change so made it variable
        k_theta=1.0 # Proportional gain for angular velocity. May want to change so made it variable

        x_c, y_c, theta_c = 0,0,msg[-1]
        x_t, y_t = msg[0], msg[1]

        if x_t == 0 and y_t == 0:
            v = 0
            omega = 0
            vel = DriveCommand()
            vel.angularvel = float(omega)
            vel.linearvel = float(v)
            self.drive_command_publisher.publish(vel)
            return

        # Compute the distance to the target
        dx = x_t - x_c
        dy = y_t - y_c
        distance = np.sqrt(dx**2 + dy**2)

        # Compute the angle to the target relative to the car's orientation
        angle_to_target = np.atan2(dy, dx)
        angle_error = angle_to_target - theta_c

        # Normalize the angle error to [-pi, pi]
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

        # Compute control outputs
        v = k_p * distance  # Linear velocity (proportional to distance)
        omega = k_theta * angle_error  # Angular velocity (proportional to angle error)

        ## Clamping velocities to the maximum values. I think this will work
        v = min(v, MAX_LINEAR)
        omega = max(min(omega, MAX_ANGULAR), -MAX_ANGULAR)

        vel = DriveCommand()
        vel.angularvel = omega
        vel.linearvel = v
        self.drive_command_publisher.publish(vel)



def main(args=None) -> None:
    """
    The entrypoint of the program, this will create the ROS2 node, tell the ROS2 runtime to 'spin' it
    and then shutdown gracefully when the node quits
    """
    rclpy.init(args=args)
    control_node = Control()
    rclpy.spin(control_node)
    rclpy.shutdown()


if __name__ == '__main__':
     main()