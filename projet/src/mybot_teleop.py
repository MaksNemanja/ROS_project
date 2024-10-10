#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import sys, termios, tty
import click
from geometry_msgs.msg import Twist
import numpy as np
from sensor_msgs.msg import LaserScan

# Arrow keys codes
keys = {'\x1b[A':'up', '\x1b[B':'down', '\x1b[C':'right', '\x1b[D':'left', 's':'stop', 'q':'quit','a':'Default','p' : 'print'}

class TeleopNode:
    def __init__(self):
        self.pub = rospy.Publisher(rospy.get_param("/cmd", default='/cmd_vel'), Twist, queue_size = 10)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.rate = rospy.Rate(10)  #10Hz
        self.emergency_stop = False
        self.linear_scale = rospy.get_param("/linear_scale", default=1.0)
        self.angular_scale = rospy.get_param("/angular_scale", default=1.0)
        self.i = 0

    def reset(self, twist):
        twist.angular.z = 0 
        twist.linear.x = 0

    def crop_vel(self, twist):
        if twist.linear.x > 2:
            twist.linear.x = 2 
        if twist.linear.x < -2:
            twist.linear.x = -2

    def scan_callback(self, data):
        ranges = np.array(data.ranges)

        if ranges[1] < 0.2:
            self.emergency_stop = True
        else:
            self.emergency_stop = False

    def run(self):
        try:
            while not rospy.is_shutdown():
                rospy.loginfo("\n\n\niter : "+str(self.i))
                self.i += 1
                twist = Twist()



                mykey = click.getchar()
                if mykey in keys.keys():
                    char = keys[mykey]

                if char == 'up':
                    twist.linear.x += 1 * self.linear_scale
                    rospy.loginfo(char)
                elif char == 'down':
                    twist.linear.x -= 1 * self.linear_scale
                    rospy.loginfo(char)
                if char == 'left':
                    twist.angular.z = 1 * self.angular_scale
                    rospy.loginfo(char)
                elif char == 'right':
                    twist.angular.z = -1 * self.angular_scale
                    rospy.loginfo(char)

                if char == "quit":
                    rospy.loginfo(char)
                    break
                if char == "stop":
                    rospy.loginfo("stooooooooooooooooooooooooooooooop")
                if char == "print":
                    print(self.i)

                rospy.loginfo(twist.linear.x)
                rospy.loginfo(self.emergency_stop)
                
                if self.emergency_stop:
                    twist.linear.x = 0
                    twist.angular.z = 0

                self.crop_vel(twist)
                self.pub.publish(twist)
                self.rate.sleep()

        except rospy.ROSInterruptException:
            pass

if __name__ == '__main__':
    rospy.init_node("mybot_teleop", anonymous=True)
    node = TeleopNode()
    node.run()

