import cv2
import time
from imutils.video import VideoStream

class Webcam:
    def __init__(self):
        self.left_vid = cv2.VideoCapture("rtsp://view:AbcdUit123@192.168.19.193")
        # self.left_vid = cv2.VideoCapture(r"G:\My Drive\MLCV_nhat_Smartlab\train_data\left-side.mp4") # link to rtsp left camera
        self.right_vid = cv2.VideoCapture("rtsp://view:AbcdUit123@192.168.19.194") # link to right camera

    def get_frame(self, is_left):
        if is_left:
            cap, img = self.left_vid.read()
        else:
            cap, img = self.right_vid.read()

        if not cap:
            print('reconnect')
            if is_left:
                self.left_vid = cv2.VideoCapture("rtsp://view:AbcdUit123@192.168.19.193")
            else:
                self.right_vid = cv2.VideoCapture("rtsp://view:AbcdUit123@192.168.19.194")

        return img