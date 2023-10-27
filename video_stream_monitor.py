import cv2
import numpy as np
from inference import Detector
import tkinter as tk
from PIL import Image, ImageTk
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import argparse


class VideoStream:
    def __init__(self,
                 quit_character: str = 'q',
                 threshold: float = .4,
                 location: str = None):
        self.video_stream_object = cv2.VideoCapture(location if location is not None else 0)
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
            fire_mask, depth_mask, coords = self.detector(image=frame[:, :, ::-1])

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quit_character', default='q', required=False)
    parser.add_argument('--threshold', default=0.4, required=False)
    parser.add_argument('--location', default=None, required=False)
    video_stream = VideoStream(**vars(parser.parse_args()))
    thread = Thread(target=video_stream.get_feed, daemon=True)
    thread.start()
    video_stream.root.mainloop()
