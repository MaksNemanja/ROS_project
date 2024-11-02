#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import CompressedImage, LaserScan, Image
from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import Twist
import numpy as np
from sensor_msgs.msg import Imu
import tf.transformations as tf
from time import time

name = 'Object detection'


white_hsv = np.array([[0, 0, 200],[45, 50, 255]]) 
yellow_hsv = np.array([[15, 50, 100],[45, 255, 255]])
green_hsv = np.array([[40, 10, 50],[80, 255, 255]])
orange_bgr = [0,127,255]
beige_bgr = [96,102,176]

class LineFollower:
    def __init__(self):
        rospy.init_node('line_follower')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image', Image, self.image_callback)# pour la simu
        #self.image_sub = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.image_callback)# pour la realité
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        
        self.line_following_enabled = True
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.d_obstacle = False
        self.left_obstacle = False
        self.right_obstacle = False
        self.last_detection = None

        self.wait_red_lines = True
        self.lim_dist = rospy.get_param('/distance_limit')
        self.tab_mean = []

        self.low_H_name = 'Low_H'
        self.low_S_name = 'Low_S'
        self.low_V_name = 'Low_V'
        self.high_H_name = 'High_H'
        self.high_S_name = 'High_S'
        self.high_V_name = 'High_V'
        self.max_value_H = 180
        self.max_value_S = 255
        self.max_value_V = 255
        self.low_H = 15
        self.low_S = 50
        self.low_V = 100
        self.high_H = 45
        self.high_S = 255
        self.high_V = 255
        self.name = "left" 
        self.low_H2 = 0
        self.low_S2 = 0
        self.low_V2 = 200
        self.high_H2 = 45
        self.high_S2 = 50
        self.high_V2 = 255 
        self.name2 = "right"

        self.nb_inf_right, self.nb_inf_left = 0, 0
        self.mean_left, self.mean_right =0, 0
        self.default_left = [0,0]
        self.default_right = [0,0]
        self.reversed = False
        self.tempo = 0

        self.compt_red_lines = rospy.get_param('/red_lines_counter')
        self.NLaunchFile = rospy.get_param('/N_launch_file')




        


    def image_callback(self,data):
        
        cmd_vel = Twist()
        """
        emergency_stop = rospy.get_param("/emergency_stop")

        while(emergency_stop):
            cmd_vel.angular.z = 0
            cmd_vel.linear.x = 0
            self.cmd_vel_pub.publish(cmd_vel)
            emergency_stop = rospy.get_param("/emergency_stop")
        """
        left_lines = None
        right_lines = None
        cv_imageCompressed = self.bridge.imgmsg_to_cv2(data,"bgr8")

        #Trackbars
        """       
        cv2.namedWindow(self.name)
        cv2.createTrackbar(self.low_H_name,self.name, self.low_H, self.max_value_H, self.on_low_H_thresh_trackbar)
        cv2.createTrackbar(self.high_H_name, self.name, self.high_H, self.max_value_H, self.on_high_H_thresh_trackbar)
        cv2.createTrackbar(self.low_S_name, self.name, self.low_S, self.max_value_S, self.on_low_S_thresh_trackbar)
        cv2.createTrackbar(self.high_S_name, self.name, self.high_S, self.max_value_S, self.on_high_S_thresh_trackbar)
        cv2.createTrackbar(self.low_V_name, self.name, self.low_V, self.max_value_V, self.on_low_V_thresh_trackbar)
        cv2.createTrackbar(self.high_V_name, self.name, self.high_V, self.max_value_V, self.on_high_V_thresh_trackbar)

        cv2.namedWindow(self.name2)
        cv2.createTrackbar(self.low_H_name,self.name2, self.low_H2, self.max_value_H, self.on_low_H_thresh_trackbar2)
        cv2.createTrackbar(self.high_H_name, self.name2, self.high_H2, self.max_value_H, self.on_high_H_thresh_trackbar2)
        cv2.createTrackbar(self.low_S_name, self.name2, self.low_S2, self.max_value_S, self.on_low_S_thresh_trackbar2)
        cv2.createTrackbar(self.high_S_name, self.name2, self.high_S2, self.max_value_S, self.on_high_S_thresh_trackbar2)
        cv2.createTrackbar(self.low_V_name, self.name2, self.low_V2, self.max_value_V, self.on_low_V_thresh_trackbar2)
        cv2.createTrackbar(self.high_V_name, self.name2, self.high_V2, self.max_value_V, self.on_high_V_thresh_trackbar2)"""
        
        # Dessiner un rectangle autour de la région d'intérêt (ROI)
        height, width = cv_imageCompressed.shape[:2]
        roi_height = int(height * 0.35)  # Ajustez la hauteur de la ROI
        roi_width = int(width * .9)    # Ajustez la largeur de la ROI
        roi_top = int(height * 0.65)     # Ajustez la position verticale de la ROI
        roi_left = int(width * 0.05)     # Ajustez la position horizontale de la ROI
        

        image = cv_imageCompressed[roi_top:roi_top+roi_height, roi_left:roi_left+roi_width]
        left_lines,cXl,cYl = self.detect_color(image,color=np.array([[self.low_H,self.low_S,self.low_V],[self.high_H,self.high_S,self.high_V]]))
        right_lines, cXr, cYr = self.detect_color(image,color=np.array([[self.low_H2,self.low_S2,self.low_V2],[self.high_H2,self.high_S2,self.high_V2]]),side="left")
        red_lines, _, _= self.detect_red(image)
        #left_lines,cXl,cYl = self.detect_color(image,color=np.array([[self.low_H,self.low_S,self.low_V],[self.high_H,self.high_S,self.high_V]]))
        #right_lines, cXr, cYr = self.detect_red(image,color=np.array([[[0, 20, 10],[160, 20, 10]],[[20, 255, 255],[180, 255, 255]]])) # Realité


        if left_lines is not None and right_lines is None:
           self.last_detection = "left"
        if left_lines is None and right_lines is not None:
           self.last_detection = "right"
           
        if red_lines is not None:
            
            self.wait_red_lines = False
            self.tempo = time()
            
        if not self.wait_red_lines and red_lines is None:
            if time() - self.tempo < 2 :
                pass
            else :
                print("red = ",self.compt_red_lines)
                self.compt_red_lines +=1
                rospy.set_param("/red_lines_counter",self.compt_red_lines)
                self.wait_red_lines = True

            

        if left_lines is not None:
        	
            for line in left_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(cv_imageCompressed, (x1+roi_left, y1+roi_top), (x2+roi_left, y2+roi_top), (0, 255, 255), 2)  # Dessiner les lignes jaunes en jaune
            #print("left_lines",len(left_lines))
        if right_lines is not None:
            for line in right_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(cv_imageCompressed, (x1+roi_left, y1+roi_top), (x2+roi_left, y2+roi_top), (255, 255, 255), 2)  # Dessiner les lignes blanches en blanc
            #print("right_lines",len(right_lines))
                
        if red_lines is not None:
            for line in red_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(cv_imageCompressed, (x1+roi_left, y1+roi_top), (x2+roi_left, y2+roi_top), (0, 0, 255), 2)  # Dessiner les lignes rouges
        
        if cXl is not None and cYl is not None and cXr is not None and cYr is not None:
               
            cX = ((cXr+cXl)/2)+roi_left
            cY = ((cYr+cYl)/2)+roi_top
            
            if cXl > cXr :
                self.reversed = True
                cX = ((self.default_right[0]+cXl)/2)+roi_left
                #if (cX < (len(cv_imageCompressed)/2)):
                #    cX = ((self.default_right[0]+cXl)/2)+roi_left
                #else :
              	#   cX = ((cXr+self.default_left[0])/2)+roi_left
            else :
                if cXr < (self.default_right[0]*3/8) or cXl > self.default_right[0]*5/8:
                    self.reversed = True
                else:
                    self.reversed = False
            if cXr < (1/2)*self.default_right[0]:
                self.reversed = True
                cX = ((self.default_right[0]+cXl)/2)+roi_left

            
            cX = int(cX)
            cY = int(cY)
            cv2.circle(cv_imageCompressed,(cX,cY),5,[50,255,50],2)
            
        if cXl is not None and cYl is not None:
            cv2.circle(cv_imageCompressed,(int(cXl+roi_left),int(cYl+roi_top)),5,[150,150,255],2)
        if cXr is not None and cYr is not None: 
            cv2.circle(cv_imageCompressed,(int(cXr+roi_left),int(cYr+roi_top)),5,[10,10,10],2)             
        #Default_left
        cv2.circle(cv_imageCompressed,(int(self.default_left[0]+roi_left),int(self.default_left[1]+roi_top)),5,orange_bgr,2)
        
        #Default_right
        cv2.circle(cv_imageCompressed,(int(self.default_right[0]+roi_left),int(self.default_right[1]+roi_top)),5,beige_bgr,2)
                 	
        #print(f"x={cX}/{len(cv_imageCompressed[0])}, y={cY}/{len(cv_imageCompressed)}")
        cv2.circle(cv_imageCompressed,(int(len(cv_imageCompressed[0])/2),int(len(cv_imageCompressed))),5,[50,50,255],2) 

        self.show(cv_imageCompressed)

        
        if self.compt_red_lines < 2 or self.compt_red_lines == 4:
            self.controlor(cmd_vel,cX,cY,dims=(int(len(cv_imageCompressed[0])/2),int(len(cv_imageCompressed))))
            
        elif self.compt_red_lines == 2:
            self.determine_move_direction(cmd_vel, left_lines, right_lines, cX, cY, dims=(int(len(cv_imageCompressed[0])/2),int(len(cv_imageCompressed))))    
            
        elif self.compt_red_lines == 3:
            self.determine_move_direction_corridor(cmd_vel, self.mean_left, self.mean_right)

        if rospy.get_param('/emergency_stop'):
            self.stop(cmd_vel=cmd_vel)

        if self.NLaunchFile == 1 and self.compt_red_lines == 3 : 
            self.stop(cmd_vel=cmd_vel) 

        if self.NLaunchFile == 2 and self.compt_red_lines == 4 :
            self.stop(cmd_vel=cmd_vel)
        if self.NLaunchFile == 4 : 
            pass

        self.cmd_vel_pub.publish(cmd_vel)   

    def stop(self,cmd_vel):
        cmd_vel.linear.x = 0
        cmd_vel.angular.z = 0   

    def controlor(self,cmd_vel,cX=50,cY=50,dims=(50,50)):
        """
        if self.reversed :
            coef = 0.5
        else:
            coef = 1"""
        x_diff = cX - dims[0]
        y_diff = cY - dims[1]
        
        norm = np.sqrt(x_diff**2 + y_diff**2)

        alpha = 40/norm
        cmd_vel.angular.z = -2*x_diff/(dims[0])


        #cmd_vel.linear.x = -0.4*y_diff*coef/(dims[1]/2)
        cmd_vel.linear.x = -0.4*y_diff*alpha**3/(dims[1]/2)
        #print(f"cmd_vel.linear.x = {cmd_vel.linear.x} , cmd_vel.angular.z = {cmd_vel.angular.z}")
        return cmd_vel
        
    def determine_move_direction(self,cmd_vel, left_lines, right_lines, cX, cY, dims=(50,50)):
        cmd_vel.linear.x = rospy.get_param('/speed_linear')       
        if self.d_obstacle:
            if self.last_detection == "left":
                cmd_vel.angular.z = -rospy.get_param('/speed_angular')
            else:
                cmd_vel.angular.z = rospy.get_param('/speed_angular')
            
        elif self.right_obstacle:
            cmd_vel.angular.z = rospy.get_param('/speed_angular')
        elif self.left_obstacle:
            cmd_vel.angular.z = -rospy.get_param('/speed_angular')
        elif left_lines is not None and right_lines is not None:
            self.controlor(cmd_vel,cX,cY, dims)
        elif left_lines is not None:
            cmd_vel.angular.z = -rospy.get_param('/speed_angular')
        elif right_lines is not None:
           cmd_vel.angular.z = rospy.get_param('/speed_angular')


         
           
    def determine_move_direction_corridor(self, cmd_vel, left_distance, right_distance):
        cmd_vel.linear.x = rospy.get_param('/speed_linear')
        if not self.nb_inf_left and not self.nb_inf_right:
            if np.abs(left_distance- right_distance)<0.2:
                None
            elif left_distance< right_distance:
                cmd_vel.angular.z = -rospy.get_param('/speed_angular')
            else:
                cmd_vel.angular.z = rospy.get_param('/speed_angular')
        elif self.nb_inf_left<self.nb_inf_right:
            cmd_vel.angular.z = -rospy.get_param('/speed_angular')
        else:
            cmd_vel.angular.z = rospy.get_param('/speed_angular')
           
        
    def scan_callback(self,data):

        ranges = np.array(data.ranges)
        concatenated_ranges = np.concatenate((ranges[-rospy.get_param('/angle_detection'):], ranges[:rospy.get_param('/angle_detection')]))      
        distance, obstacle = np.min(concatenated_ranges), np.argmin(concatenated_ranges)
        
        
        self.mean_left = np.mean(concatenated_ranges[60:])
        self.mean_right = np.mean(concatenated_ranges[:40])
        self.nb_inf_left = np.sum(np.isinf(concatenated_ranges[60:]))
        self.nb_inf_right = np.sum(np.isinf(concatenated_ranges[:40]))
        
        if rospy.get_param('/angle_detection') - 10 <= obstacle <= rospy.get_param('/angle_detection') + 10  and distance < self.lim_dist:   
            self.d_obstacle = True
        else:
            self.d_obstacle = False
            
        if rospy.get_param('/angle_detection') + 11<= obstacle <= rospy.get_param('/angle_detection') * 2 - 1 and distance < self.lim_dist:   
            self.left_obstacle = True
        else:
            self.left_obstacle = False
            
        if 0<= obstacle <= rospy.get_param('/angle_detection') - 11 and distance < self.lim_dist:   
            self.right_obstacle = True
        else:
            self.right_obstacle = False

        if ranges[0] < rospy.get_param('/distance_critique') and not (np.isinf(ranges[0]) or ranges[0] < 0.001):
            rospy.set_param('/emergency_stop',True)
        else:
            rospy.set_param('/emergency_stop',False)

        

        

            

    def detect_red(self,image):

        # Convertir l'image de la ROI en niveaux de gris
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Convertir l'image de la ROI en espace de couleur HSV
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Définir les limites de la plage de couleur jaune dans l'espace HSV
        lower_red0 = np.array([5, 50, 90])
        upper_red0 = np.array([15, 255, 255])
        lower_red1 = np.array([150, 50, 90])
        upper_red1 = np.array([180, 255, 255])

        # Créer un masque pour les lignes rouges dans la ROI
        red_mask0 = cv2.inRange(image_hsv, lower_red0, upper_red0)
        red_mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
        red_mask = cv2.bitwise_or(red_mask0,red_mask1)
        
        # Appliquer le masque à l'image en niveaux de gris de la ROI
        masked_gray_roi_image = cv2.bitwise_and(image_gray, image_gray, mask=red_mask)
        
        # Appliquer un flou gaussien pour réduire le bruit
        blurred_mask = cv2.GaussianBlur(masked_gray_roi_image, (3, 3), 0)
        
        # Appliquer une transformation morphologique pour améliorer la détection des lignes
        kernel = np.ones((2, 2), np.uint8)
        morphed_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_CLOSE, kernel)

        cX,cY = self.moment(morphed_mask,default=[len(image_gray[0]),len(image_gray)/2])
        # Appliquer la détection de lignes à l'image masquée de la ROI
        red_lines = cv2.HoughLinesP(masked_gray_roi_image, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        return red_lines,cX,cY
     

    def detect_color(self,image,color=np.array([[0,0,0],[0,0,0]]),side = "right"):
        

        # Convertir l'image de la ROI en niveaux de gris
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Convertir l'image de la ROI en espace de couleur HSV
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Définir les limites de la plage de couleur a gauche dans l'espace HSV
        lower_white = color[0]
        upper_white = color[1]

        # Créer un masque pour les lignes jaunes dans la ROI
        left_mask = cv2.inRange(image_hsv, lower_white, upper_white)

        # Appliquer le masque à l'image en niveaux de gris de la ROI
        masked_gray_roi_image = cv2.bitwise_and(image_gray, image_gray, mask=left_mask)
        
        # Appliquer un flou gaussien pour réduire le bruit
        blurred_mask = cv2.GaussianBlur(masked_gray_roi_image, (3, 3), 0)
        
        # Appliquer une transformation morphologique pour améliorer la détection des lignes
        kernel = np.ones((2, 2), np.uint8)
        morphed_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_CLOSE, kernel)

        if side == "right":
            self.default_right = [len(image_gray[0]),len(image_gray)/2]
            default = self.default_right
            
        elif side == "left":
            self.default_left = [0,len(image_gray)/2]
            default= self.default_left

        cX,cY = self.moment(morphed_mask,default)
        # Appliquer la détection de lignes à l'image masquée de la ROI
        left_lines = cv2.HoughLinesP(morphed_mask, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        return left_lines,cX,cY    

    def show(self,image,title="Detected Lines"):
        #bloc test

        cv2.imshow(title, image)
        cv2.waitKey(1)  

    def moment(self,morphed_mask,default):

        ret,thresh = cv2.threshold(morphed_mask,127,255,0)
        
        M = cv2.moments(thresh)
        if round(M["m00"]) == 0:
            cX = default[0]
            cY = default[1]
            
        else:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        return cX,cY


    def run(self):
        rospy.spin()
        print("bruh")

    def on_high_H_thresh_trackbar(self,val):
        self.high_H = val
        self.high_H = max(self.high_H, self.low_H+1)
        cv2.setTrackbarPos(self.high_H_name, name, self.high_H)


    def on_low_H_thresh_trackbar(self,val):
        self.low_H = val
        self.low_H = min(self.high_H-1, self.low_H)
        cv2.setTrackbarPos(self.low_H_name, name, self.low_H)


    def on_high_S_thresh_trackbar(self,val):
        self.high_S = val
        self.high_S = max(self.high_S, self.low_S+1)
        cv2.setTrackbarPos(self.high_S_name, name, self.high_S)
    
    
    def on_low_S_thresh_trackbar(self,val):
        self.low_S = val
        self.low_S = min(self.low_S-1, self.low_S)
        cv2.setTrackbarPos(self.high_S_name, name, self.low_S)

    def on_high_V_thresh_trackbar(self,val):
        self.high_V = val
        self.high_V = max(self.high_V, self.low_V+1)
        cv2.setTrackbarPos(self.high_V_name, name, self.high_V)
    
    def on_low_V_thresh_trackbar(self,val):
        self.low_V = val
        self.low_V = min(self.low_V-1, self.low_V)
        cv2.setTrackbarPos(self.high_V_name, name, self.low_V)

    # Seconde trackbar
    def on_high_H_thresh_trackbar2(self,val):
        self.high_H2 = val
        self.high_H2 = max(self.high_H2, self.low_H2+1)
        cv2.setTrackbarPos(self.high_H_name, name, self.high_H2)


    def on_low_H_thresh_trackbar2(self,val):
        self.low_H2 = val
        self.low_H2 = min(self.high_H2-1, self.low_H2)
        cv2.setTrackbarPos(self.low_H_name, name, self.low_H2)


    def on_high_S_thresh_trackbar2(self,val):
        self.high_S2 = val
        self.high_S2 = max(self.high_S2, self.low_S2+1)
        cv2.setTrackbarPos(self.high_S_name, name, self.high_S2)
    
    
    def on_low_S_thresh_trackbar2(self,val):
        self.low_S2 = val
        self.low_S2 = min(self.low_S2-1, self.low_S2)
        cv2.setTrackbarPos(self.high_S_name, name, self.low_S2)

    def on_high_V_thresh_trackbar2(self,val):
        self.high_V2 = val
        self.high_V2 = max(self.high_V2, self.low_V2+1)
        cv2.setTrackbarPos(self.high_V_name, name, self.high_V2)
    
    def on_low_V_thresh_trackbar2(self,val):
        self.low_V2 = val
        self.low_V2 = min(self.low_V2-1, self.low_V2)
        cv2.setTrackbarPos(self.high_V_name, name, self.low_V2)

if __name__ == '__main__':
    try:

        line_follower = LineFollower()
        line_follower.run()
    except rospy.ROSInterruptException:
        pass
