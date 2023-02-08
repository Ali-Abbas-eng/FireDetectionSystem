import cv2
import numpy as np

from models import UNet
import torchvision.transforms as transform
import os


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
                 fire_detection_model_path: str = os.path.join('Models', 'UNet Checkpoints'), ip: str = None,
                 checkpoint: int = None):
        """
        Class constructor

        Parameters:
        ----------
        quit_character: str, optional (default = 'q')
            Character used to stop the video stream.
        fire_detection_model_path: str, optional (default = os.path.join('Models', 'UNet Checkpoints'))
            Path to the directory containing the U-Net model checkpoints.
        ip: str, optional (default = None)
            IP address of the camera to be used for video stream. If None, default camera will be used.
        checkpoint: int, optional (default = None)
            Index of the checkpoint to be used.

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
        self.model = UNet(model_directory=fire_detection_model_path, checkpoint_index=checkpoint)

        # Define the image pre-processing pipeline
        self.frame_pre_processing_pipeline = transform.Compose([
            # Convert the image to a PIL image
            transform.ToPILImage(),
            # Crop the image to a 256x256 size
            transform.CenterCrop((256, 256)),
            # Convert the image to a tensor
            transform.ToTensor()
        ])

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

            # Display the processed frame
            cv2.imshow('Video Stream', self.process(frame))

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
        complete_frame = frame.copy()

        # Use the U-Net model to generate a fire detection mask
        mask = self.model.infer(self.frame_pre_processing_pipeline(frame).unsqueeze(dim=0))

        # Calculate the pixel coordinates of the center of the mask and the complete frame
        frame_pixel_starts = mask.shape[0] // 2, mask.shape[1] // 2
        height_start, width_start = complete_frame.shape[0] // 2, complete_frame.shape[1] // 2

        # Overlay the mask on the complete frame
        complete_frame[height_start - frame_pixel_starts[0]: height_start + frame_pixel_starts[0],
                       width_start - frame_pixel_starts[1]: width_start + frame_pixel_starts[1], :] = mask

        # Return the processed frame
        return complete_frame


if __name__ == '__main__':
    x = VideoStream()
    x.get_feed()
