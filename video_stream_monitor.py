import cv2
import numpy as np
from inference import Detector
import tkinter as tk
from PIL import Image, ImageTk
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib import cm


class VideoStream:
    def __init__(self, quit_character: str = 'q',
                 threshold: float = .4,
                 ip: str = None):
        self.video_stream_object = cv2.VideoCapture(f'https://{ip}:8080/video' if ip is not None else 0)
        self.quit_character = quit_character
        self.detector = Detector()
        self.threshold = threshold

        # Initialize Tkinter viewing functionality
        self.root = tk.Tk()
        self.frame_label = tk.Label(self.root)
        self.frame_label.grid(row=0, column=0)
        self.fire_mask_label = tk.Label(self.root)
        self.fire_mask_label.grid(row=1, column=0)
        self.depth_mask_label = tk.Label(self.root)
        self.depth_mask_label.grid(row=1, column=1)
        self.text_label = tk.Label(self.root, text="Random text")
        self.text_label.grid(row=0, column=1)
        self.normaliser = plt.Normalize(vmin=self.detector.depth_estimator.min_depth,
                                        vmax=self.detector.depth_estimator.max_depth)

    def get_feed(self):
        capture = True
        while capture:
            _, frame = self.video_stream_object.read()
            if frame is None:
                continue
            frame = cv2.resize(frame, (640, 480))
            # New Version:
            fire_mask, depth_mask = self.process(frame=frame)

            # Send to Tkinter viewing functionality
            frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_photo = ImageTk.PhotoImage(image=frame_image)
            self.frame_label.config(image=frame_photo)
            self.frame_label.image = frame_photo

            fire_mask_image = Image.fromarray(cv2.cvtColor(fire_mask, cv2.COLOR_BGR2RGB))
            fire_mask_photo = ImageTk.PhotoImage(image=fire_mask_image)
            self.fire_mask_label.config(image=fire_mask_photo)
            self.fire_mask_label.image = fire_mask_photo

            depth_mask_rgb = cm.plasma(self.normaliser(depth_mask[:, :, 0]))
            depth_mask_rgb = np.uint8(depth_mask_rgb * 255)
            depth_map_image = Image.fromarray(depth_mask_rgb)
            depth_map_photo = ImageTk.PhotoImage(image=depth_map_image)

            self.depth_mask_label.config(image=depth_map_photo)
            self.depth_mask_label.image = depth_map_photo
            # depth_mask_image = Image.fromarray(cv2.cvtColor(depth_mask, cv2.COLOR_BGR2RGB))
            # depth_mask_photo = ImageTk.PhotoImage(image=depth_mask_image)
            # self.depth_mask_label.config(image=depth_mask_photo)
            # self.depth_mask_label.image = depth_mask_photo

            if cv2.waitKey(1) & 0xFF == ord(self.quit_character):
                capture = False
        self.video_stream_object.release()
        cv2.destroyAllWindows()

    def process(self, frame):
        """
        Processes a single frame of the video stream using the U-Net model to generate a fire detection mask.

        Args:
            frame (np.ndarray): A single frame of the video stream.

        Returns:
            np.ndarray: The processed frame with the fire detection mask overlayed on the original frame.
        """
        # Use the U-Net model to generate a fire detection mask
        pre_processed_frame = (np.copy(frame)[None, :, :, ::-1]).astype(np.float16)

        # Generate the fire and depth masks using the detector
        fire_mask, depth_mask = self.detector(pre_processed_frame)

        # Apply a threshold to the fire mask to create a binary mask
        fire_mask = np.where(fire_mask > self.threshold, 255, 0).astype(np.uint8)
        fire_mask = fire_mask[0]

        # Extract the depth mask for the first frame
        depth_mask = (depth_mask[0][0])[:, :, None]

        # Detect fire in the frame using the fire and depth masks
        frame = self.detect_fire(frame=frame, fire_mask=fire_mask, depth_mask=depth_mask)
        depth_mask = ((depth_mask - np.min(depth_mask)) / (np.max(depth_mask) - np.min(depth_mask)) * 255).astype(
            'uint8')
        # Return the processed frame
        return fire_mask, depth_mask

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

        # Create a new mask with the same shape as the frame
        mask = np.zeros_like(frame)

        # Draw the contours on the new mask
        [cv2.drawContours(mask, contours__, -1, 255, thickness=-1) for _ in contours__]

        # Find the points where the new mask is white (i.e. where there is fire)
        points = np.where(mask == 255)

        # If there are no points where there is fire, return the original frame
        if len(points) == 0:
            return frame

        # Calculate the average depth of the points where there is fire
        average_depth = np.mean(depth_mask[points])

        # Add text to the frame indicating the average depth of the fire
        cv2.putText(frame,
                    text=f'Fire {average_depth:.03f} meters',
                    org=(0, frame.shape[0]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.4,
                    color=(0, 0, 255),
                    thickness=1)

        # Return the modified frame
        return frame


if __name__ == '__main__':
    video_stream = VideoStream(ip='10.181.138.155')
    thread = Thread(target=video_stream.get_feed)
    thread.start()
    video_stream.root.mainloop()
