import numpy as np
import csv
import cv2
from tflite_runtime.interpreter import Interpreter
from nms import nms


class PalmDetection:
    def __init__(self, palm_model_path, anchors_path):
        self.interp_palm = Interpreter(palm_model_path)
        self.interp_palm.allocate_tensors()

        output_details = self.interp_palm.get_output_details()
        input_details = self.interp_palm.get_input_details()

        self.in_idx = input_details[0]['index']
        self.out_reg_idx = output_details[0]['index']
        self.out_clf_idx = output_details[1]['index']
        # reading the SSD anchors
        with open(anchors_path, "r") as csv_f:
            self.anchors = np.r_[
                [x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]
            ]

        # 90° rotation matrix used to create the alignment trianlge
        self.R90 = np.r_[[[0, 1], [-1, 0]]]

        # trianlge target coordinates used to move the detected hand
        # into the right position
        self._target_triangle = np.float32([
            [128, 128],
            [128, 0],
            [0, 128]
        ])
        self._target_box = np.float32([
            [0, 0, 1],
            [256, 0, 1],
            [256, 256, 1],
            [0, 256, 1],
        ])

    @staticmethod
    def _sigm(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _im_normalize(img):
        return np.ascontiguousarray(
            2 * ((img / 255) - 0.5
                 ).astype('float32'))

    def preprocess_img(self, img):
        # fit the image into a 256x256 square
        shape = np.r_[img.shape]
        self.rgb_shape = shape
        pad = (shape.max() - shape[:2]).astype('uint32') // 2
        img_pad = np.pad(
            img,
            ((pad[0], pad[0]), (pad[1], pad[1]), (0, 0)),
            mode='constant')
        img_small = cv2.resize(img_pad, (256, 256))
        img_small = np.ascontiguousarray(img_small)

        img_norm = self._im_normalize(img_small)
        return img_pad, img_norm, pad

    def predict_hand_boxes(self, img_norm, pad):
        self.interp_palm.set_tensor(self.in_idx, img_norm.reshape(1, 256, 256, 3))
        self.interp_palm.invoke()
        out_reg = self.interp_palm.get_tensor(self.out_reg_idx)[0]  # bbox
        out_clf = self.interp_palm.get_tensor(self.out_clf_idx)[0, :, 0]  # scores

        detecion_mask = self._sigm(out_clf) > 0.7
        print(np.sum(detecion_mask))
        candidate_detect = out_reg[detecion_mask]
        candidate_anchors = self.anchors[detecion_mask]

        if candidate_detect.shape[0] == 0:
            return None

        keep = nms(candidate_detect, 0.5)
        bboex = []

        for idx in keep:
            dx, dy, w, h = candidate_detect[idx, :4]
            center_wo_offst = candidate_anchors[idx, :2] * 256
            dx += center_wo_offst[0] - pad[1]
            dy += center_wo_offst[1] - pad[0]
            bboex.append((dx, dy, w, h))

        print('keep', keep)
        return bboex

    def __call__(self, img):
        img_pad, img_norm, pad = self.preprocess_img(img)
        return self.predict_hand_boxes(img_norm, pad)
