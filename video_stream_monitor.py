import cv2
import numpy as np

from models import get_depth_estimation_model, get_fire_segmentation_model
import os
import matplotlib.pyplot as plt
from GLOBAL_VARIABLES import MAX_DEPTH, MIN_DEPTH


def de_normalise_depth(average_depth):
    return (MAX_DEPTH - MIN_DEPTH) * average_depth + MIN_DEPTH


class VideoStream:
    """
    Class for video stream processing

    This class processes a video stream and performs fire detection using a model. The video stream can be obtained either
    from a webcam or IP-based camera. The model used is a U-Net, which is imported from the 'models' module. The processed
    video stream can be displayed and stopped by pressing a specified quit character.

    Attributes:
    ----------
    video_stream_object: cv2.VideoCapture
        Object to capture the video stream.
    quit_character: str
        Character used to stop the video stream.
    model: UNet
        U-Net model for fire detection.
    frame_pre_processing_pipeline: torchvision.transforms.Compose
        A pipeline of transformations to apply on the video frame before feeding it to the model.

    Methods:
    -------
    get_feed()
        Captures and processes the video stream until the quit character is pressed.
    process(frame: numpy.ndarray)
        Processes a single frame from the video stream.
    """

    def __init__(self, quit_character: str = 'q',
                 threshold: float = .4,
                 ip: str = None):
        """
        Class constructor

        Parameters:
        ----------
        quit_character: str, optional (default = 'q')
            Character used to stop the video stream.

        threshold: float, optional (default = 0.4)
            confidence value on which we split the output map to fire/non-fire pixels

        ip: str, optional (default = None)
            IP address of the camera to be used for video stream. If None, default camera will be used.

        Raises:
        ------
        AssertionError
            If the length of quit_character is not 1.
        """
        # Validate that the quit_character input has a length of 1
        assert len(quit_character) == 1, "you can only choose one character to be used as a quit command"
        # Initialize the video stream
        self.video_stream_object = cv2.VideoCapture(f'https://{ip}:8080/video' if ip is not None else 0)

        # Store the quit_character input
        self.quit_character = quit_character

        # Load the U-Net model from the provided path and checkpoint
        self.fire_segmentation_model = get_fire_segmentation_model(assert_file_exist=True)
        self.depth_estimation_model = get_depth_estimation_model(assert_file_exist=True)
        self.threshold = threshold

    def get_feed(self):
        """
        This function captures and displays video frames from the video stream. The video stream can be stopped by pressing
        the quit character defined in the constructor.

        Returns:
        -------
        None
        """
        # Set capture flag to True to start the video stream
        capture = True

        # Capture and display video frames until the quit character is pressed
        while capture:
            # Read a frame from the video stream
            _, frame = self.video_stream_object.read()

            # Check if the frame is None and skip if it is
            if frame is None:
                continue

            # calculate the processed frame
            frame = self.process(frame=frame)
            cv2.imshow('Video Stream', frame)

            # Check if the quit character has been pressed
            if cv2.waitKey(1) & 0xFF == ord(self.quit_character):
                # If the quit character has been pressed, set the capture flag to False to stop the video stream
                capture = False

        # Release the video stream
        self.video_stream_object.release()

        # Destroy all windows created by OpenCV
        cv2.destroyAllWindows()

    def process(self, frame):
        """
        This function processes a single frame of the video stream. The frame is processed using the U-Net model, which
        outputs a fire detection mask. The mask is then overlayed on the original frame and returned.

        Parameters:
        ----------
        frame: numpy.ndarray
            A single frame of the video stream.

        Returns:
        -------
        numpy.ndarray
            The processed frame with the fire detection mask overlayed on the original frame.
        """
        # Copy the frame to avoid modifying the original frame
        # complete_frame = frame.copy()
        # Use the U-Net model to generate a fire detection mask
        pre_processed_frame = (np.copy(frame)[None, :, :, ::-1] / 255.).astype(np.float16)
        fire_mask = self.fire_segmentation_model(pre_processed_frame)
        fire_mask = np.where(fire_mask > self.threshold, 255, 0).astype(np.uint8)
        fire_mask = fire_mask[0]
        depth_mask = self.depth_estimation_model(pre_processed_frame)
        depth_mask = depth_mask[0]
        depth_mask = np.array(depth_mask)
        frame = self.detect_fire(frame=frame, fire_mask=fire_mask, depth_mask=depth_mask)
        # Return the processed frame
        return frame[:, :, ::-1]

    @staticmethod
    def detect_fire(frame, fire_mask, depth_mask):
        frame = np.where(fire_mask > 0, frame, frame // 3)

        average_depth = 0

        contours__, __ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_x, text_y = [0, 0]
        x, y, width, height = None, None, None, None
        mask = np.zeros_like(frame)
        for __ in contours__:
            cv2.drawContours(mask, contours__, -1, 255, thickness=-1)
            c = max(contours__, key=cv2.contourArea)
            x, y, width, height = cv2.boundingRect(c)
            text_x = x + width
            text_y = y
        points = np.where(mask == 255)
        if len(points) == 0:
            return frame
        average_depth = np.mean(depth_mask[points])
        average_depth = de_normalise_depth(average_depth)
        cv2.putText(frame,
                    text=f'Fire {average_depth:.02f} meters',
                    org=(text_x, text_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.3,
                    color=(0, 0, 255),
                    thickness=1)
        return frame[:, :, ::-1]


if __name__ == '__main__':
    x = VideoStream(threshold=.6)
    x.get_feed()
