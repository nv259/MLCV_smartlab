from webcam import Webcam
from yolov7.yolo_detect import YoloDetect
import cv2
import time

webcam = Webcam()
detector = YoloDetect()


def read_from_webcam(is_left):
    count = 0
    while True:
        # Read image from class Webcam
        image = next(webcam.get_frame((is_left)))

        if image is None:
            continue

        t1 = time.time()
        img, ccl, ccr = detector.detect(image, is_left, count)
        t2 = time.time()

        cv2.imshow(img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    read_from_webcam(1)