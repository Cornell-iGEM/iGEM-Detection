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

#GPIO.cleanup()

#pin numbers on pi for LEDs
excite_low_pin = 18
GPIO.setup( excite_low_pin, GPIO.OUT)
excite_high_pin = 23
GPIO.setup( excite_high_pin, GPIO.OUT)
pdawn_pin = 20
GPIO.setup( pdawn_pin, GPIO.OUT)

GPIO.setup(32, GPIO.OUT)
GPIO.output(32, False)
GPIO.setup(5, GPIO.OUT)
GPIO.output(5, False)

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
camera.framerate = 1
camera.shutter_speed = 320000*10 #for darker environments
#camera.shutter_speed = 3200*3 #light testing

pwm = GPIO.PWM(18, 1000)
pwm.stop()
pwm.start(1)
redLower = np.array((0,60, 40))
#redLower = np.array((15,60, 50))
redUpper = np.array((330, 255,255))
#redUpper = np.array((60, 255, 255))

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
        if radius > 1: 
            #draw the circle and centroid on the frame, 
            #then update the list of tracked points 
            cv2.circle(frame, (int(x), int(y)), int(radius), 
                                    (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
        #cv2.imshow('c', frame)
        #key = cv2.waitKey(0) & 0xFF
        #return img_integral/area
        return img_integral, bounds, area #green light reflection is localized, shouldn't normalize
    # show the frame to our screen
    



    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = frame
    return -1, -1

csvfile = open('baseline-42.csv', 'wb')
try:
#make function which takes in frame, lower and uppper bound for hue saturation value, return integral 
    fieldnames = ['emission1', 'x1', 'x2', 'y1', 'y2', 'dutycycle', 'area', 'time']
    
    csvwriter  = csv.DictWriter(csvfile, fieldnames=fieldnames)
    csvwriter.writeheader()
    count = 0
    while True:
        #for dc in (20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 33, 50, 67, 75, 100):
        count +=1
        for dc in (42,):
        #response = raw_input("ledsample")
        #if response == "q":
            #    break
            
        #print('block test 1')
        #low excitation
            #GPIO.output( excite_low_pin, GPIO.HIGH)
            pwm.ChangeDutyCycle(dc)
            #GPIO.output(excite_low_pin, GPIO.HIGH)
            time.sleep(0.5)

            camera.capture(raw_capture, format='bgr')
            frame = raw_capture.array     
            #if count % 203 == 0:
                #cv2.imwrite(str(time.ctime()) +"-"+ str(count) + "-" + str(dc)+"-raw", frame)
            results1 = brightnessvalue(frame, redLower, redUpper)
            x = results1[0]

            if x < 0:
                print(results1)
                raw_capture.truncate(0)
                continue
            bounds1 = results1[1]
            area = results1[2]
            ##GPIO.output( excite_low_pin, GPIO.LOW)
            raw_capture.truncate(0)
            '''
            #high excitation
            #take new picture
            GPIO.output( excite_high_pin, GPIO.HIGH)
            time.sleep(0.1)
            camera.capture(raw_capture, format='bgr')
            frame = raw_capture.array
            results2 = brightnessvalue(frame, redLower, redUpper)  
            y = results2[0]
            bounds2 = results2[1]
            GPIO.output( excite_high_pin, GPIO.LOW)
            raw_capture.truncate(0)
            if x != 0 and y != 0: 
            ratio = x/y
            else:
            ratio = -1
            '''
            data = {"emission1": x, "x1": bounds1[0], "x2": bounds1[1], "y1": bounds1[2], "y2": bounds1[3], "time": time.ctime(), "dutycycle": dc, 'area': area}
            csvwriter.writerow(data)
            csvfile.flush()
                                                                
        #url = 'http://citronnade.mooo.com/rfp'
            print(data)
       # requests.post(url, data=data)
        

              
        


finally:
    cv2.destroyAllWindows()
    camera.close()
    pwm.stop()
    #pwm.stop()
    GPIO.cleanup()
    csvfile.close()
