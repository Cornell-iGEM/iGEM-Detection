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


#pwm = GPIO.PWM(18, 100)
#pwm.start(100)

while raw_input('Enter to send a pulse.  q to quit.') != 'q':
    print('hi')
    GPIO.output(18, GPIO.HIGH)
    time.sleep(1)
    GPIO.output(18, GPIO.LOW)
    print('bye')



#pwm.stop()
GPIO.cleanup()
