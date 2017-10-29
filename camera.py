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
#GPIO.setmode(GPIO.BCM)
from fractions import Fraction
#
#GPIO.setup(18, GPIO.OUT)




"""
# initialize the camera
cam = VideoCapture(0)   # 0 -> index of camera
s, img = cam.read()
if s:    # frame captured without any errors
    '''namedWindow("cam-test",cv2.CV_WINDOW_AUTOSIZE)
    imshow("cam-test",img)
    waitKey(0)
    destroyWindow("cam-test")'''
    imwrite("filename.jpg",img) #save image

"""

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
#camera.awb_gains = (Fraction(2), Fraction(2))
for video_frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    frame = video_frame.array
    
    
    

#cap = cv2.VideoCapture(0)
    #pwm = GPIO.PWM(18, 50)
    #pwm.start(8)
#pwm.on()

#while(True):
    # Capture frame-by-frame
    #ret, frame = cap.read()
    
    #print camera.awb_gains
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = frame

    
    integral_table = cv2.integral(frame)
    image_y = int(frame.shape[0])
    image_x = int(frame.shape[1])

#    print image_x

    #    print (integral_table[image_y][image_x] + integral_table[0][0] - integral_table[0][image_x] - integral_table[image_y][0])
    #avg_value = integral_table[image_y][image_x][0] / (image_y*image_x)

    #upper right quadrant
    
    avg_value_1 = (integral_table[0][int(image_x/2)][0] + integral_table[int(image_y/2)][image_x][0] - integral_table[int(image_y/2)][int(image_x/2)][0] - integral_table[0][image_x][0]) / (image_y*image_x / 4.0)
    avg_value_2 = (integral_table[image_y/2][int(image_x/2)][0] + integral_table[0][0][0] - integral_table[int(image_y/2)][0][0] - integral_table[0][image_x/2][0]) / (image_y*image_x / 4.0)
    avg_value_3 = (integral_table[int(image_y)][int(image_x/2)][0] + integral_table[int(image_y/2)][0][0] - integral_table[int(image_y/2)][int(image_x/2)][0] - integral_table[image_y][0][0]) / (image_y*image_x / 4.0)
    avg_value_4 = (integral_table[image_y][int(image_x)][0] + integral_table[int(image_y/2)][int(image_x/2)][0] - integral_table[int(image_y)][int(image_x/2)][0] - integral_table[int(image_y/2)][image_x][0]) / (image_y*image_x / 4.0)
    
    quadrant_intensity = [(avg_value_1, 1), (avg_value_2, 2), (avg_value_3, 3), (avg_value_4, 4)]
    quadrant_intensity.sort(key = lambda x:int(x[0]), reverse=True)
    #print quadrant_intensity

    #print (avg_value_1)
    quadrant_no = quadrant_intensity[0][1]
    #print 'Quadrant ' + str(quadrant_no) + ' is the most intense'
    #print quadrant_intensity[quadrant_no-1][0] * 100/255
    
    #pwm.ChangeDutyCycle(int(avg_value_1 * 100/255))
    
    quadrant_center =(int(image_x/4) + int(image_x/2 * (quadrant_no == 1 or quadrant_no == 4) ) ,int(image_y/4) + int(image_y/2 * (quadrant_no > 2)) ) 
    #print 'Quadrant center is at ' + str(quadrant_center)
    cgray = cv2.medianBlur(gray, 5)

    #cv2.circle(cgray, quadrant_center, 10, (255,255,255), -1)
    cv2.circle(frame, quadrant_center, 10, (255,255,255), -1)

    

    #cv2.imshow('frame',frame)

    #encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
    

    
    #ret = cv2.imencode('.jpg', cgray, buf)#, encode_param)
    cv2.imwrite("temp.jpg", frame)

    with open("temp.jpg", "rb") as content:
        #jpeg_im = content.read()
        files = {'media': content}
        brightness = {"brightness": quadrant_intensity[quadrant_no-1][0] * 100/255}
        url = 'http://10.42.0.1:5000/upload_data'
        url2 = 'http://10.42.0.1:5000/brightness'
        #url = 'http://citronnade.mooo.com/upload'
        #url2 = 'http://citronnade.mooo.com/brightness'
        requests.post(url, files=files)
        requests.post(url2, data=brightness)
        
    #key = cv2.waitKey(30) & 0xFF
    time.sleep(0.02)
    # clear the stream in preparation for the next frame
    raw_capture.truncate(0)

    #if the `q` key was pressed, break from the loop
    #if key == ord("q"):
     #   break
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

# When everything done, release the capture
#cap.release()
cv2.destroyAllWindows()
#pwm.stop()
GPIO.cleanup()
#f.close()
