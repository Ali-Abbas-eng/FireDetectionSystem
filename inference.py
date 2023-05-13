import os
from fire_detection.network import get_fire_segmentation_model
from AdaBins.infer import InferenceHelper
import numpy as np
from threading import Thread
import cv2


class Detector:
    def __init__(self, threshold: float = .4):
        self.fire_detector = get_fire_segmentation_model(os.path.join('fire_detection', 'models'), 'model.h5')
        self.depth_estimator = InferenceHelper(dataset='nyu')
        self.fire_mask = None
        self.depth_mask = None
        self.threshold = threshold

    def get_fire_mask(self, image):
        self.fire_mask = self.fire_detector(image)

    def get_depth_mask(self, image):
        _, self.depth_mask = self.depth_estimator.predict_pil(np.transpose(image, [0, 3, 1, 2]))

    def __call__(self, image):
        image = image[None] / 255.
        fire_thread = Thread(target=self.get_fire_mask, args=(image, ))
        depth_thread = Thread(target=self.get_depth_mask, args=(image, ))
        fire_thread.start()
        depth_thread.start()

        # Wait for both threads to finish executing using threading.Thread.join() method.
        fire_thread.join()
        depth_thread.join()

        return self.post_process(self.fire_mask, self.depth_mask)

    def post_process(self, fire_mask, depth_mask):
        """
               Processes a single frame of the video stream using the U-Net model to generate a fire detection mask.

               Args:
                   frame (np.ndarray): A single frame of the video stream.

               Returns:
                   np.ndarray: The processed frame with the fire detection mask overlayed on the original frame.
               """

        # Apply a threshold to the fire mask to create a binary mask
        fire_mask = np.where(fire_mask > self.threshold, 255, 0).astype(np.uint8)
        fire_mask = fire_mask[0]

        # Find the contours of the fire mask
        contours__, __ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_contour = []
        for contour in contours__:
            if len(contour) > len(max_contour):
                max_contour = contour

        # If there are no points where there is fire, return the original frame
        if len(max_contour) == 0:
            # Return the processed frame
            return fire_mask, depth_mask[0][0], (None, None, None)

        coordinates = max_contour[len(max_contour) // 2][0]

        # Calculate the average depth of the points where there is fire
        average_depth = np.min(depth_mask.flatten())
        # Return the processed frame
        return fire_mask, depth_mask[0][0], (*coordinates, average_depth)

    @staticmethod
    def detect_fire(frame, fire_mask, depth_mask):
        """
        Detects fire in a given frame using a fire mask and a depth mask.

        Args:
            frame (np.ndarray): The frame in which to detect fire.
            fire_mask (np.ndarray): The mask indicating the presence of fire in the frame.
            depth_mask (np.ndarray): The mask indicating the depth of each pixel in the frame.

        Returns:
            np.ndarray: The frame with fire detection information added.
        """
        # Darken the areas of the frame where there is no fire
        frame = np.where(fire_mask > 0, frame, frame // 3)

        # Find the contours of the fire mask
        contours__, __ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_contour = []
        for contour in contours__:
            if len(contour) > len(max_contour):
                max_contour = contour

        # Find the points where the new mask is white (i.e. where there is fire)
        points = max_contour

        # If there are no points where there is fire, return the original frame
        if len(points) == 0:
            return frame

        # Calculate the average depth of the points where there is fire
        average_depth = np.mean(depth_mask[points])

        # Return the modified frame
        return average_depth
