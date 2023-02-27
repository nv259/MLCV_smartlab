import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture("rtsp://mmlab:mmlab@2022@192.168.19.10:554/ch01/0")
    while True:
        _, img = cap.read()
        cv2.imshow('test', img)

    cv2.destroyAllWindows()