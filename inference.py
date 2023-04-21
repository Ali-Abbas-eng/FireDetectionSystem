import os
from fire_detection.network import get_fire_segmentation_model
from AdaBins.infer import InferenceHelper
import numpy as np
from PIL import Image
from matplotlib import cm


class Detector:
    def __init__(self):
        self.fire_detector = get_fire_segmentation_model(os.path.join('fire_detection', 'models'), 'model.h5')
        self.depth_estimator = InferenceHelper(dataset='nyu')

    def __call__(self, image):
        image = image / 255.
        fire_mask = self.fire_detector(image)
        image = Image.fromarray(image[0].astype('uint8'), 'RGB')
        dist_mask = self.depth_estimator.predict_pil(image)[1]
        return fire_mask, dist_mask
