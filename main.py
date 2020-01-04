import numpy as np
import sys
import cv2
from SingleHandTrack import SingleLandmark
from PalmDetection import PalmDetection

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2
landmark_model_path = 'model/hand_landmark.tflite'
palm_detection_model_path = 'model/palm_detection_without_custom_op.tflite'
anchors_path = 'model/anchors.csv'
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]


def landmark(rgb_img):
    points = ldmark(rgb_img)
    points = points.astype(np.int)

    for x, y in points:
        cv2.circle(img, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
    for connection in connections:
        x0, y0 = points[connection[0]]
        x1, y1 = points[connection[1]]
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
    cv2.imshow("landmark", img)
    cv2.waitKey(1)


def detection(img):
    bboxes = palm(rgb_img)
    img = cv2.resize(img, (256, 256))

    if bboxes is not None:

        for point in bboxes:
            cv2.line(img, (int(point[0]), int(point[1])), (int(point[0] + point[2]), int(point[1])), (0, 0, 255), 2)
            cv2.line(img, (int(point[0] + point[2]), int(point[1])),
                     (int(point[0] + point[2]), int(point[1] + point[3])), (0, 0, 255), 2)
            cv2.line(img, (int(point[0]), int(point[1] + point[3])),
                     (int(point[0] + point[2]), int(point[1] + point[3])), (0, 0, 255), 2)
            cv2.line(img, (int(point[0]), int(point[1] + point[3])), (int(point[0]), int(point[1])), (0, 0, 255), 2)

    cv2.imshow("palm", img)
    cv2.waitKey(1)


if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    if cam.isOpened() != True:
        print("USB Camera Open Error!!!")
        sys.exit(0)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    ldmark = SingleLandmark(landmark_model_path)
    palm = PalmDetection(palm_detection_model_path, anchors_path)
    while cam.isOpened():
        s, img = cam.read()
        if not s:
            break
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmark(rgb_img)
