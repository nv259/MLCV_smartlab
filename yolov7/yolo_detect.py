import sys
sys.path.insert(0, './yolov7')
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
from utils.ROI import left_camera, right_camera, draw_polygon


def draw_text(img, text,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=3,
              font_thickness=2,
              text_color=(0, 255, 0),
              text_color_bg=(0, 0, 0)
              ):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

class YoloDetect():
    def __init__(self, weights=r'./yolov7/trained__pt/best.pt', img_size=384, iou_thres=0.45, conf_thres=0.25):
        # self.source = source
        self.weights = weights
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.img_size = img_size
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        # self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
        #     ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Initialize
        set_logging()
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.img_size, s=self.stride)  # check img_size

        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1

    def detect(self, im0s, is_left=True, count=0):

        t0 = time.time()
        # for path, img, im0s, vid_cap in dataset:
        img = letterbox(im0s, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (
                self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img)[0]
        t2 = time_synchronized()
        print("Inference time:", t2-t1)

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # My dictionary to count device
            ccl = dict()
            ccl['keyboard'] = 0
            ccl['mouse'] = 0
            ccl['tv'] = 0

            ccr = dict()
            ccr['keyboard'] = 0
            ccr['mouse'] = 0
            ccr['tv'] = 0

            s, im0 = '', im0s

            # Draw ROI
            if is_left == 1:
                left_points, right_points = left_camera()
            else:
                left_points, right_points = right_camera()

            im0 = draw_polygon(im0, left_points)
            im0 = draw_polygon(im0, right_points)

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    check = plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1,
                                         is_left=is_left)
                    if check != 0:
                        _class = self.names[int(cls)]

                        if check == -1:
                            ccl[_class] = ccl[_class] + 1
                        else:
                            ccr[_class] = ccr[_class] + 1

                fps = 1000 / (1E3 * (t2 - t1) + (1E3 * (t3 - t2)))
                fps = int(fps)
                fps = str(fps)

                sl = 'left-ROI:   ' + str(ccl)
                sr = 'right-ROI:  ' + str(ccr)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(im0, fps, (7, 1000), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
                # cv2.putText(im0, s, (7, 70), font, 2, (100, 255, 0), 3, cv2.LINE_AA)
                draw_text(im0, sl, font, (7, 70), 2, 2, (100, 255, 0))
                draw_text(im0, sr, font, (7, 140), 2, 2, (100, 255, 0))
            # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        # print(f'Done. ({time.time() - t0:.3f}s)')
        return im0, ccl, ccr
