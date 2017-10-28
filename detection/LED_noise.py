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
import csv

def integral(x1, x2, y1, y2, table):
    return table[y1][x1][0] + table[y2][x2][0] - table[y1][x2][0] - table[y2][x1][0]

#pin numbers on pi for LEDs
excite_low_pin = 18
GPIO.setup( excite_low_pin, GPIO.OUT)
excite_high_pin = 23
GPIO.setup( excite_high_pin, GPIO.OUT)
pdawn_pin = 20
GPIO.setup( pdawn_pin, GPIO.OUT)

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
camera.awb_gains = (Fraction(5,4), Fraction(4,3))
#camera.shutter_speed = 32000 #for darker environments
camera.shutter_speed = 3200*3 #light testing

#pwm = GPIO.PWM(18, 100)
#pwm.start(1)
redLower = np.array((0,50, 150))
redUpper = np.array((330, 255,255))

def brightnessvalue(frame, redLower, redUpper):
    #Avisha: ball tracking
    #print('block test 2')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    #cv2.imshow('gr', frame)
    #key = cv2.waitKey(0) & 0xFF
    #construct mask, dilations and erosions to remove noise 
    mask = cv2.inRange(hsv, redLower, redUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    #find contours in the mask, initialize current center (x,y)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    b,g,r = cv2.split(frame)
    b = cv2.bitwise_and(b, mask)
    g = cv2.bitwise_and(g, mask)
    r = cv2.bitwise_and(r, mask)
    frame = cv2.merge((b,g,r))
    averagemask = cv2.mean(frame, mask= mask)
    integral_table = cv2.integral(frame) 
    image_y = int(frame.shape[0])
    image_x = int(frame.shape[1])
    #cv2.imshow('gr', frame)
    #key = cv2.waitKey(0) & 0xFF
        #only proceed if at least one contour was found 
    if len (cnts) > 0:
                    #find largest contour, use it to compute min enclosed cirlce
                    #and centroid 
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        bounds = max(0, x -radius), min(image_x-1, x + radius), max(0, y - radius), min(image_y-1, y + radius)
        #print(bounds)
        img_integral = integral(bounds[0], bounds[1], bounds[2], bounds[3], integral_table)
        #img_integral = integral(0, image_x, 0, image_y, integral_table)
        area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
        #print(img_integral/area)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        #proceed if radius is min size --NEED TO FIGURE OUT
        #if radius > 1: 
            #draw the circle and centroid on the frame, 
            #then update the list of tracked points 
        #    cv2.circle(frame, (int(x), int(y)), int(radius), 
        #                            (0, 255, 255), 2)
        #    cv2.circle(frame, center, 5, (0, 0, 255), -1)
        return img_integral/area
    # show the frame to our screen
    



    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = frame
    return 0

csvfile = open('LED.csv', 'wb')
try:
#make function which takes in frame, lower and uppper bound for hue saturation value, return integral 
    fieldnames = ['emission1', 'emission2', 'time']
    
    csvwriter  = csv.DictWriter(csvfile, fieldnames=fieldnames)
    csvwriter.writeheader()
    while True:
        #response = raw_input("ledsample")
        #if response == "q":
        #    break
       
        #print('block test 1')
        #low excitation
        GPIO.output( excite_low_pin, GPIO.HIGH)
        time.sleep(0.1)
        camera.capture(raw_capture, format='bgr')
        frame = raw_capture.array     
        x = brightnessvalue(frame, redLower, redUpper)
        GPIO.output( excite_low_pin, GPIO.LOW)
        raw_capture.truncate(0)
        #high excitation
        #take new picture
        GPIO.output( excite_high_pin, GPIO.HIGH)
        time.sleep(0.1)
        camera.capture(raw_capture, format='bgr')
        frame = raw_capture.array
        y = brightnessvalue(frame, redLower, redUpper)  
        GPIO.output( excite_high_pin, GPIO.LOW)
        raw_capture.truncate(0)
        if x != 0 and y != 0: 
            ratio = x/y
        else:
            ratio = -1

        data = {"emission1": x, "emission2": y, "time": time.ctime()}
        csvwriter.writerow(data)
        csvfile.flush()
        #url = 'http://citronnade.mooo.com/rfp'
        print(data)
       # requests.post(url, data=data)
        

              
        


finally:
    cv2.destroyAllWindows()
    camera.close()
    #pwm.stop()
    GPIO.cleanup()
    csvfile.close()
