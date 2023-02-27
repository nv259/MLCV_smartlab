import cv2
import time
from imutils.video import VideoStream

class Webcam:
    def __init__(self):
        # self.left_vid = cv2.VideoCapture(0)
        # self.left_vid = VideoStream("rtsp://mmlab:mmlab@2022@192.168.19.10:554/ch01/0")
        self.left_vid = cv2.VideoCapture("rtsp://view:AbcdUit123@192.168.19.193")
        # self.left_vid = cv2.VideoCapture(r"G:\My Drive\MLCV_nhat_Smartlab\train_data\left-side.mp4") # link to rtsp left camera
        # self.right_vid = cv2.VideoCapture("rtsp://view:AbcdUit123@192.168.19.194") # link to right camera

    def get_frame(self, is_left):

        while True:
            if is_left:
                _, img = self.left_vid.read()
            else:
                _, img = self.right_vid.read()

            if img is None:
                continue

            yield img#cv2.imencode('.jpg', img)[1].tobytes()
            time.sleep(1/30)
