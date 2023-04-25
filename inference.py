import os
from fire_detection.network import get_fire_segmentation_model
from AdaBins.infer import InferenceHelper
from PIL import Image
from threading import Thread


class Detector:
    def __init__(self):
        self.fire_detector = get_fire_segmentation_model(os.path.join('fire_detection', 'models'), 'model.h5')
        self.depth_estimator = InferenceHelper(dataset='nyu')
        self.fire_mask = None
        self.depth_mask = None

    def get_fire_mask(self, image):
        self.fire_mask = self.fire_detector(image / 255.)

    def get_depth_mask(self, image):
        image = Image.fromarray(image[0].astype('uint8'), 'RGB')
        _, self.depth_mask = self.depth_estimator.predict_pil(image)

    def __call__(self, image):
        fire_thread = Thread(target=self.get_fire_mask, args=(image, ))
        depth_thread = Thread(target=self.get_depth_mask, args=(image, ))
        fire_thread.start()
        depth_thread.start()

        # Wait for both threads to finish executing using threading.Thread.join() method.
        fire_thread.join()
        depth_thread.join()

        return self.fire_mask, self.depth_mask