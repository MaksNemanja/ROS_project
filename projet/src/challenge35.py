#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import Twist
import numpy as np
import math
from sensor_msgs.msg import Imu
import tf.transformations as tf
from nav_msgs.msg import Odometry

class Doors():
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('doors')
        # Initialize the CvBridge object
        self.bridge = CvBridge()
        # Subscribe to the camera image topic
        self.image_sub = rospy.Subscriber('/camera/image', Image, self.image_callback)
        # Publisher for velocity commands
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # Get parameters for linear and angular speed
        self.speed_linear = rospy.get_param('/speed_linear')
        self.speed_angular = rospy.get_param('/speed_angular')
        # Get parameters for angle detection and distance limit
        self.angle_detection = rospy.get_param('/angle_detection')
        self.distance_limit_bottle = rospy.get_param('/distance_limit_bottle')
        # Subscribe to the laser scan topic
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        # Initialize steps and flags
        self.first_step, self.second_step, self.third_step, self.fourth_step = False, False, False, False
        # Set the rate for the ROS loop
        self.rate = rospy.Rate(10)
        # Initialize obstacle flags
        self.obstacles = [False, False, False]
        # Initialize accomplishment counter
        self.accomplished = 0
        # Subscribe to odometry topic
        self.pose_sub = rospy.Subscriber('/odom', Odometry, self.pose_callback)
        # Initialize variable for orientation
        self.pos_z = None
        # Flag to track the first detected bottle
        self.first_bottle = False
        # Time variable to store initial orientation
        self.t1 = None
    
    def pose_callback(self, msg):
        # Extract orientation (in quaternions) from odometry
        orientation = msg.pose.pose.orientation
        # Convert quaternions to Euler angles (roll, pitch, yaw)
        euler_angles = tf.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        _, _, yaw = euler_angles
        yaw_normalized = yaw % (2 * math.pi)
        self.pos_z = yaw_normalized
        
    def image_callback(self, data):
        # Convert ROS image message to OpenCV format
        image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        # Define region of interest (ROI) for drawing a rectangle
        height, width = image.shape[:2]
        roi_height = int(height * 0.25)
        roi_width = int(width * 0.16)
        roi_top = int(height * 0.40)
        roi_left = int(width * 0.41)
        # Draw a rectangle around the ROI
        cv2.rectangle(image, (roi_left, roi_top), (roi_left + roi_width, roi_top + roi_height), (255, 255, 255), 2)
        # Extract ROI from the image
        roi_image = image[roi_top:roi_top+roi_height, roi_left:roi_left+roi_width]
        # Detect blue, green, and red contours in the ROI
        blue_contours = self.detect_color_contours(roi_image, np.array([90, 50, 50]), np.array([130, 255, 255]))
        green_contours = self.detect_color_contours(roi_image, np.array([40, 50, 50]), np.array([80, 255, 255]))
        red_contours = self.detect_red_contours(roi_image)
        # Update obstacle flags based on contour detection
        self.obstacles[0] = len(blue_contours) != 0
        self.obstacles[1] = len(green_contours) != 0
        self.obstacles[2] = len(red_contours) != 0
        # Draw filtered contours on the original image
        output = image.copy()
        if self.accomplished == 0:
            cv2.drawContours(output, blue_contours, -1, (0, 255, 0), 2)
        elif self.accomplished == 1:
            cv2.drawContours(output, green_contours, -1, (0, 255, 0), 2)
        elif self.accomplished == 2:
            cv2.drawContours(output, red_contours, -1, (0, 255, 0), 2)
        # Display the resulting image
        cv2.imshow("Filtered Contours", output)
        cv2.waitKey(1)
        
    def detect_color_contours(self, image, lower_color, upper_color):
        # Convert image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Threshold the image to get pixels in specified color range
        mask = cv2.inRange(hsv, lower_color, upper_color)
        # Apply blur and erosion/dilation to clean the mask
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter contours by size
        min_contour_area = 500
        max_contour_area = 5000
        filtered_contours = []
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if min_contour_area < contour_area < max_contour_area:
                filtered_contours.append(contour)
        return filtered_contours
        
    def detect_red_contours(self, image):
        # Convert image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Define lower and upper bounds for red color range in HSV space
        lower_red0 = np.array([5, 50, 90])
        upper_red0 = np.array([15, 255, 255])
        lower_red1 = np.array([150, 50, 90])
        upper_red1 = np.array([180, 255, 255])
        # Create masks for red lines in the ROI
        red_mask0 = cv2.inRange(hsv, lower_red0, upper_red0)
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask = cv2.bitwise_or(red_mask0, red_mask1)
        # Apply blur and erosion/dilation to clean the mask
        mask = cv2.GaussianBlur(red_mask, (5, 5), 0)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter contours by size
        min_contour_area = 500
        max_contour_area = 5000
        filtered_contours = []
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if min_contour_area < contour_area < max_contour_area:
                filtered_contours.append(contour)
        return filtered_contours
        
    def rotate(self, angle, angular_speed, direction):
        twist = Twist()
        # Convert average angle to radians
        angle_radians = math.radians(angle)
        # Set rotation direction
        direction = 1 if direction == "left" else -1
        # Calculate duration for rotation
        duration_seconds = abs(angle_radians / angular_speed)
        # Initialize start time
        initial_time = rospy.Time.now().to_sec()
        # Publish rotation commands
        while not rospy.is_shutdown():
            current_time = rospy.Time.now().to_sec()
            elapsed_time = current_time - initial_time
            if elapsed_time < duration_seconds:
                # Continue rotating
                twist.angular.z = angular_speed * direction
                self.cmd_vel_pub.publish(twist)
            else:
                # Stop rotation
                self.cmd_vel_pub.publish(twist)
                break
            self.rate.sleep()
            
    def scan_callback(self, data):
        # Process laser scan data to detect walls
    
        # Convert laser scan data to numpy array
        ranges = np.array(data.ranges)
    
        # Concatenate ranges to handle wrap-around at 0 degrees
        concatenated_ranges1 = np.concatenate((ranges[-self.angle_detection:], ranges[:self.angle_detection]))
    
        # Find the minimum distance and its corresponding angle
        distance1, angle1 = np.min(concatenated_ranges1), np.argmin(concatenated_ranges1)
    
        # Initialize Twist message for velocity commands
        cmd_vel = Twist()
    
        # Step 1: Move forward or rotate based on obstacle detection
        if not self.first_step:
            if not self.obstacles[self.accomplished]:
                # No obstacle detected, rotate
                cmd_vel.angular.z = self.speed_angular
            else:
                # Obstacle detected, move forward
                cmd_vel.linear.x = self.speed_linear
            
            # Check if a bottle is detected within a certain range
            if self.obstacles[self.accomplished] and np.any(concatenated_ranges1[self.angle_detection - 20:self.angle_detection + 20] < self.distance_limit_bottle):
                # Store current orientation as initial orientation
                self.t1 = self.pos_z
                # Set flag to indicate the first bottle detection
                self.first_bottle = True
                # Move to the next step
                self.first_step = True
    
        # Step 2: Rotate until a second bottle is detected
        if self.first_step and not self.second_step:
            if self.first_bottle or not self.obstacles[self.accomplished]:
                # Rotate if no obstacle or first bottle detected
                cmd_vel.angular.z = self.speed_angular
            else:
                # Move to the next step if obstacle is cleared
                self.second_step = True
            # Reset first bottle flag if no obstacle is detected
            if not self.obstacles[self.accomplished]:
                self.first_bottle = False
            
        # Step 3: Move forward or rotate based on obstacle detection
        if self.second_step and not self.third_step:
            if not self.obstacles[self.accomplished]:
                # No obstacle detected, rotate
                cmd_vel.angular.z = self.speed_angular
            else:
                # Obstacle detected, move forward
                cmd_vel.linear.x = self.speed_linear
            
            # Check if a bottle is detected within a certain range
            if self.obstacles[self.accomplished] and np.any(concatenated_ranges1[self.angle_detection - 20:self.angle_detection + 20] < self.distance_limit_bottle):
                # Move to the next step if bottle is detected
                self.third_step = True
            
        # Step 4: Rotate to align with the next wall
        t2 = self.pos_z
        if self.third_step and not self.fourth_step:
            # Calculate the angle to rotate based on current and initial orientations
            if t2 > self.t1:
                angular = np.abs(t2 - self.t1) * (180/math.pi)
            else:
                angular = (2*np.pi - self.t1 + t2) * (180/math.pi)
            # Rotate halfway to the next wall
            if angular < 180:
                self.rotate(angular/2, self.speed_angular, "right")
            else:
                self.rotate((360-angular)/2, self.speed_angular, "left")
            # Move to the next step
            self.fourth_step = True
        
        # Step 5: Move forward and adjust orientation based on wall proximity
        if self.fourth_step:
            cmd_vel.linear.x = self.speed_linear
            # Adjust orientation based on wall proximity
            if distance1 < 0.15:
                if self.angle_detection - 65 < angle1 < self.angle_detection:
                    cmd_vel.linear.x = 0.0
                    cmd_vel.angular.z = self.speed_angular
                elif self.angle_detection < angle1 < self.angle_detection + 65:
                    cmd_vel.linear.x = 0.0
                    cmd_vel.angular.z = -self.speed_angular
            # succesful step if no obstacles are detected within the distance limit
            if not np.any(ranges < self.distance_limit_bottle):    
                self.accomplished += 1
                self.first_bottle = False
                # if the challenge is not completed, we start the steps again
                if self.accomplished !=3:
                    self.first_step, self.second_step, self.third_step, self.fourth_step = False, False, False, False
                # else, challenge completed
                else:
                    None
             
        # Publish the velocity command
        self.cmd_vel_pub.publish(cmd_vel)


    def run(self):
        rospy.spin()
        
if __name__ == '__main__':
    try:
        doors = Doors()        
        doors.run()
    except rospy.ROSInterruptException:
        pass

