import cv2
import numpy as np
from inference import Detector
import tkinter as tk
from PIL import Image, ImageTk
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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

        self.text_label = tk.Label(self.root, text="Initialising Display ....")
        self.text_label.grid(row=0, column=1)
        self.normaliser = plt.Normalize(vmin=self.detector.depth_estimator.min_depth,
                                        vmax=self.detector.depth_estimator.max_depth)
        self.fig, self.ax = plt.subplots(1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=1, column=1)
        # self.depth_mask_label = tk.Label(self.root)
        # self.depth_mask_label.grid(row=1, column=1)

    def get_feed(self):
        capture = True
        while capture:
            _, frame = self.video_stream_object.read()
            if frame is None:
                continue
            frame = cv2.resize(frame, (640, 480))
            # New Version:
            fire_mask, depth_mask, coords = self.process(frame=frame)

            # Send to Tkinter viewing functionality
            frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_photo = ImageTk.PhotoImage(image=frame_image)
            self.frame_label.config(image=frame_photo)
            self.frame_label.image = frame_photo

            fire_mask_image = Image.fromarray(cv2.cvtColor(fire_mask, cv2.COLOR_BGR2RGB))
            fire_mask_photo = ImageTk.PhotoImage(image=fire_mask_image)
            self.fire_mask_label.config(image=fire_mask_photo)
            self.fire_mask_label.image = fire_mask_photo

            # depth_mask_rgb = cm.plasma(self.normaliser(depth_mask[:, :, 0]))
            plt.axis('off')
            self.ax.imshow(depth_mask, cmap='plasma')
            self.canvas.draw()
            self.canvas.flush_events()
            self.fig.canvas.flush_events()

            self.text_label.configure(text=f'Fire at:\n\tx = {coords[0]}\n\ty = {coords[1]}\n\tz = {coords[2]}')
            self.text_label.update()
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


if __name__ == '__main__':
    video_stream = VideoStream()
    # video_stream = VideoStream(ip='192.168.126.231')
    thread = Thread(target=video_stream.get_feed, daemon=True)
    thread.start()
    video_stream.root.mainloop()
