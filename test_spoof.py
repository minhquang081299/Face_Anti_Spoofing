import os
import cv2
import numpy as np
# import argparse
import warnings
import time
# import imutils

from imutils.video import VideoStream
from test_realtime import detect_spoofing
# frame = './Face_Anti_Spoofing/3.png'
cap = VideoStream().start()

while True:
    frame = cap.read()
    image = cv2.flip(frame, 1)
    speed_test = 0
    time_p = time.time()
    if detect_spoofing(image):
        print("Real")
        # speed_test = time.time() - time_p
        # print("Time_test", speed_test)
    else:
        print("Spoof")

    speed_test = time.time() - time_p
    print("Time_test:", speed_test)
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()