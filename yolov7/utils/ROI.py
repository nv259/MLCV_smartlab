import cv2
import numpy as np


def left_camera():
  left_points = [ [350, 602], [866, 396], [1558, 862], [1434, 1050], [480, 1051], [350, 602] ]
  right_points = [ [955, 366], [1274, 236], [1852, 547], [1779, 876], [955, 366] ]
  return left_points, right_points

def right_camera():
  left_points = [ [585, 260], [906, 376], [540, 557], [270, 380], [585, 260] ]
  right_points = [ [1024, 380], [1508, 550], [1250, 945], [625, 600], [1024, 380] ]
  return left_points, right_points

def draw_polygon(frame, points):
    for point in points:
        frame = cv2.circle(frame, (point[0], point[1]), 5, (0, 0, 255), -1)
    
    frame = cv2.polylines(frame, [np.int32(points)], False, (255, 0, 0), thickness=1)
    
    return frame