import cv2
import cv2.cv as cv
import numpy as np
import signal, os, subprocess, sys
import time
import threading
import requests
import io

from picamera.array import PiRGBArray
from picamera import PiCamera

import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
from fractions import Fraction
#
GPIO.setup(18, GPIO.OUT)



camera = PiCamera()
camera.framerate = 32

#camera.framerate = Fraction(1,6)
raw_capture = PiRGBArray(camera)
output = PiRGBArray(camera)
time.sleep(0.1)
"""
#g = camera.awb_gains
g = (Fraction(1, 1), Fraction(1,1))
print g
camera.exposure_mode = 'off'
camera.shutter_speed = 500000

camera.awb_mode = 'off'
camera.awb_gains = g
camera.capture(output, format="bgr")
img = output.array
b,g,r = cv2.split(img)
cv2.imshow('frame',g)
key = cv2.waitKey(0) & 0xFF
"""

camera.awb_mode = 'off'
camera.awb_gains = (Fraction(1,3), Fraction(1,3))
camera.shutter_speed = 32000

pwm = GPIO.PWM(18, 100)
pwm.start(1)
redLower = np.array((0, 127, 58))
redUpper = np.array((330, 255,255))
#camera.awb_gains = (Fraction(2), Fraction(2))
try:
    for video_frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        frame = video_frame.array
        #Avisha: ball tracking
        cv2.imshow('asdf', frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #construct mask, dilations and erosions to remove noise 
        mask = cv2.inRange(hsv, redLower, redUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
              
            #find contours in the mask, initialize current center (x,y)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
             
            #only proceed if at least one contour was found 
        if len (cnts) > 0:
                        #find largest contour, use it to compute min enclosed cirlce
                        #and centroid 
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                        #proceed if radius is min size --NEED TO FIGURE OUT
            if radius > 1: 
                                #draw the circle and centroid on the frame, 
                                #then update the list of tracked points 
                cv2.circle(frame, (int(x), int(y)), int(radius), 
                                        (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                #update the points queue 
           
            
                #loop over the set of tracked points 
           
                        
                        
 
            # show the frame to our screen
        #cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF


        #print camera.awb_gains
        print (float(camera.awb_gains[0]), float(camera.awb_gains[1]))
        print (camera.exposure_speed)
        # gains are about 1/3, 1/3
        # Our operations on the frame come here
               #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = frame


        #integral_table = cv2.integral(frame)
        image_y = int(frame.shape[0])
        image_x = int(frame.shape[1])

        #cv2.imshow('temp', frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        time.sleep(0.02)
        # clear the stream in preparation for the next frame
        raw_capture.truncate(0)


finally:
    cv2.destroyAllWindows()
    camera.close()
    pwm.stop()
    GPIO.cleanup()

