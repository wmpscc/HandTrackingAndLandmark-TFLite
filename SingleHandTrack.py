import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter


class SingleLandmark:
    def __init__(self, landmark_model_path):
        self.interp_joint = Interpreter(landmark_model_path)
        self.interp_joint.allocate_tensors()
        self.in_idx_joint = self.interp_joint.get_input_details()[0]['index']
        self.out_idx_joint = self.interp_joint.get_output_details()[0]['index']

    def predict_joint(self, img_norm):
        self.interp_joint.set_tensor(self.in_idx_joint, img_norm.reshape(1, 256, 256, 3))
        self.interp_joint.invoke()
        joints = self.interp_joint.get_tensor(self.out_idx_joint)
        return joints.reshape(-1, 2)

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

    def __call__(self, img):
        img_pad, img_norm, pad = self.preprocess_img(img)
        joints = self.predict_joint(img_norm)
        scale = 256. / self.rgb_shape.max()
        points = joints / scale
        points[:, 0] = points[:, 0] - pad[1] - 1
        points[:, 1] = points[:, 1] - pad[0] - 1
        return points
