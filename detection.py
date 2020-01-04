import argparse
import io
import re
import sys
import time

from annotation import Annotator

import numpy as np
import cv2

from PIL import Image
from tflite_runtime.interpreter import Interpreter

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    # scores = get_output_tensor(interpreter, 2)
    # count = int(get_output_tensor(interpreter, 3))
    print(boxes.shape)
    print(classes.shape)
    results = []

    return results


def annotate_objects(annotator, results, labels):
    """Draws the bounding box and label for each object in the results."""
    for obj in results:
        # Convert the bounding box figures from relative coordinates
        # to absolute coordinates based on the original resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * CAMERA_WIDTH)
        xmax = int(xmax * CAMERA_WIDTH)
        ymin = int(ymin * CAMERA_HEIGHT)
        ymax = int(ymax * CAMERA_HEIGHT)

        # Overlay the box, label, and score on the camera preview
        annotator.bounding_box([xmin, ymin, xmax, ymax])
        annotator.text([xmin, ymin],
                       '%s\n%.2f' % (labels[obj['class_id']], obj['score']))


def main():
    path_model = "./model/palm_detection_without_custom_op.tflite"
    path_labels = "./model/palm_detection_labelmap.txt"
    threshold = 0.4

    labels = load_labels(path_labels)

    interpreter = Interpreter(path_model)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    cam = cv2.VideoCapture(0)
    if cam.isOpened() != True:
        print("USB Camera Open Error!!!")
        sys.exit(0)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    while cam.isOpened():
        s, img = cam.read()
        if not s:
            break
        img = cv2.resize(img, (256, 256))
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detect_objects(interpreter, rgb_img, threshold)
        print(results)

    # with picamera.PiCamera(
    #         resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=30) as camera:
    #     camera.start_preview()
    #     try:
    #         stream = io.BytesIO()
    #         annotator = Annotator(camera)
    #         for _ in camera.capture_continuous(
    #                 stream, format='jpeg', use_video_port=True):
    #             stream.seek(0)
    #             image = Image.open(stream).convert('RGB').resize(
    #                 (input_width, input_height), Image.ANTIALIAS)
    #             start_time = time.monotonic()
    #             results = detect_objects(interpreter, image, threshold)
    #             elapsed_ms = (time.monotonic() - start_time) * 1000
    #
    #             annotator.clear()
    #             annotate_objects(annotator, results, labels)
    #             annotator.text([5, 0], '%.1fms' % (elapsed_ms))
    #             annotator.update()
    #
    #             stream.seek(0)
    #             stream.truncate()
    #
    #     finally:
    #         camera.stop_preview()


if __name__ == '__main__':
    main()
