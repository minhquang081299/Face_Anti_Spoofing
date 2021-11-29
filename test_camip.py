import os
import cv2
import numpy as np
# import argparse
import warnings
import time

import sklearn
from imutils.video import VideoStream
from test_realtime import detect_spoofing

# cap = VideoStream().start()
from sklearn.datasets import images

cap = cv2.VideoCapture(1)
cap.open("rtsp://192.168.1.15:554/Streaming/Channels/001")
while True:
    ret, frame = cap.read()
    image = cv2.flip(frame, 1)
# speed_test = 0
# time_p = time.time()

    # if detect_spoofing(image):
    #     print("Real")
    #     # speed_test = time.time() - time_p
    #     # print("Time_test", speed_test)
    # else:
    #     print("Spoof")

# speed_test = time.time() - time_p
# print("Time_test:", speed_test)
    cv2.imshow('Face Recognition', image)
    # cv2.imwrite('test.jpg', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
# cap.release()
# cv2.destroyAllWindows()