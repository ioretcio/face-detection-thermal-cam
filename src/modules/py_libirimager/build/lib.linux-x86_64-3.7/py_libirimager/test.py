import sys
import numpy as np
import cv2
from ir_cam import IrCamera
with IrCamera("ircam_config.xml".encode('utf-8')) as ir_cam:
    for i in range(500):
        data_t, data_p, _ = ir_cam.get_frame()
        data_p -= 1200
        print(data_p)
        cv2.imshow('visual data', data_p)
        cv2.waitKey(5)

