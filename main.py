import time
import numpy as np

from flask import Flask, render_template, Response
from webcam import Webcam
from yolov7.yolo_detect import YoloDetect
import cv2


app = Flask(__name__)
webcam = Webcam()
device_detector = YoloDetect(r'./yolov7/trained__pt/devices.pt')
# human_detector = YoloDetect(r'./yolov7/trained__pt/human.pt')
ccl = dict()
ccr = dict()


@app.route("/")
def index():
    return render_template("index.html")

def read_from_webcam(is_left):
    count = 0
    while True:
        # Read image from class Webcam
        image = webcam.get_frame(is_left)

        if image is None:
            continue

        # Detect using Yolov7
        t1 = time.time()
        img, ccl, ccr = device_detector.detect(image, is_left, count)
        t2 = time.time()

        #TODO: i think we should inference per 5 frames instead
        img = cv2.resize(image, (1280, 720))
        img = cv2.imencode('.jpg', img)[1].tobytes()

        # Return image to web by yield cmd
        yield b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n--frame\r\n'

@app.route("/left_camera")
def left_camera():
    return Response( read_from_webcam(1), mimetype="multipart/x-mixed-replace; boundary=frame" )

@app.route("/right_camera")
def right_camera():
    return Response( read_from_webcam(0), mimetype="multipart/x-mixed-replace; boundary=frame" )

#TODO: web sockets
#TODO: using AJAX to dynamically update statements (monitor, mouse, keyboard)
@app.route('/statement')
def statement():
    return 0

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)