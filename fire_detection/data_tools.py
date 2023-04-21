import os
import random
from zipfile import ZipFile
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
from skimage import measure, draw
from detectron2.structures import BoxMode
import json
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
import argparse
from PIL import Image
from typing import Callable
import tensorflow as tf
from abc import ABC


def create_record(image_path: str, mask_path: str, file_id: int, special_mask_process: Callable) -> Tuple[bool, Dict]:
    """
    Create a record for an image and its corresponding mask in a custom dataset for detectron2.

    This function reads an image and its corresponding mask file and creates a record containing
    information about the image such as its file name, id, height and width. It also processes the
    mask image to find contours of each instance in the binary mask and creates annotations for
    each instance including its bounding box coordinates and segmentation information.

    Args:
        image_path (str): The file name of the image.
        mask_path (str): The file name of the mask image.
        file_id (int): The id of the image.
        special_mask_process (Callable): the function that knows how to understand the mask image.

    Returns:
        dict: A dictionary containing information about the image and its annotations.
    """
    try:
        # Read the input image using matplotlib imread function
        image = Image.open(image_path)
        image = np.array(image)

        # Initialize a record dictionary with basic information about the input image
        record = {"file_name": image_path,
                  "image_id": file_id,
                  "height": image.shape[0],
                  "width": image.shape[1]}

        # Read mask_image using matplotlib imread function and convert it to binary format
        mask_image = special_mask_process(mask_path)

        # Find contours of each instance in the binary mask using skimage find_contours function
        contours = measure.find_contours(mask_image, 0.5)

        # Initialize a list to store annotations for instances in the input image
        objs = []

        # Iterate over each contour found in the binary mask
        for contour in contours:
            # Create a boolean mask for the current instance with same shape as input binary_mask
            instance_mask = np.zeros_like(mask_image, dtype=np.uint8)

            # Use skimage draw.polygon function to draw polygon around current contour
            rr, cc = draw.polygon(contour[:, 0], contour[:, 1])

            # Set values inside polygon to 1
            instance_mask[rr, cc] = 1

            # Find bounding box coordinates of current instance by finding min/max x/y values where value is equal to 1
            pos = np.where(instance_mask == 1)
            x_min = int(np.min(pos[1]))
            x_max = int(np.max(pos[1]))
            y_min = int(np.min(pos[0]))
            y_max = int(np.max(pos[0]))

            points_x = pos[1]
            points_y = pos[0]

            poly = [(x + 0.5, y + 0.5) for x, y in zip(points_x, points_y)]

            poly = [int(p) for x in poly for p in x]

            if len(poly) < 8:
                continue

            # Create annotation dictionary for current instance containing bounding box coordinates,
            # segmentation information and other details
            obj = {
                "bbox": [x_min, y_min, x_max, y_max],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0,
            }

            objs.append(obj)

        record["annotations"] = objs

        return True, record
    except ValueError:
        return False, {}


def extract(zip_file: os.PathLike,
            output_dir: os.PathLike,
            image_mark: str,
            mask_mark: str,
            pre_process: Callable,
            output_info_file: os.PathLike,
            mask_processor: Callable,
            train_split: float):
    """Extracts images and masks from a zip file and saves them to an output directory.

    Args:
        zip_file (os.PathLike): The path to the zip file containing the images and masks.
        output_dir (os.PathLike): The path to the directory where the extracted files will be saved.
        image_mark (str): A string used to identify image files in the zip file.
        mask_mark (str): A string used to identify mask files in the zip file.
        pre_process (Callable): A function used to pre-process the extracted images and masks.
        output_info_file (os.PathLike): The path to the file where the extracted data information will be saved.
        mask_processor (Callable): A function used to process the extracted masks.
        train_split (float): The fraction of the data to use for training.

    Returns:
        None
    """
    # Create a list to store information about the extracted data
    dataset_dicts = []

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the zip file
    with ZipFile(zip_file, 'r') as file_handle:
        # Find all image and mask files in the zip file
        images = list(sorted([member for member in file_handle.namelist() if image_mark in member]))
        masks = list(sorted([member for member in file_handle.namelist() if mask_mark in member]))

        # Extract and pre-process each image and mask
        index = 0
        for i in tqdm(range(len(images))):
            member = images[i]
            image_member = file_handle.extract(member=member, path=output_dir)
            pre_process(image_member)
            mask_member_name = masks[i]
            mask__member = file_handle.extract(mask_member_name, path=output_dir)
            pre_process(mask__member)

            # Create a record for the extracted data
            result, data_record = create_record(image_member, mask__member, index, mask_processor)
            if result:
                dataset_dicts.append(data_record)
                index += 1

    # Shuffle the data and split it into training and testing sets
    random.shuffle(dataset_dicts)
    train = dataset_dicts[:int(len(dataset_dicts) * train_split)]
    test = dataset_dicts[len(train):]

    # Update the image IDs in the training and testing sets
    def regulate_id(data):
        """
        Regulates the image IDs in a given data set.

        Args:
            data (list): The data set to regulate.

        Returns:
            None
        """
        # Iterate over the data set and update the image IDs
        for i in range(len(data)):
            data[i]['image_id'] = i

    regulate_id(train)
    regulate_id(test)

    # Save the training and testing sets to JSON files
    json.dump(train, open(os.path.join(output_dir, output_info_file + '_train.json'), 'w'))
    json.dump(test, open(os.path.join(output_dir, output_info_file + '_test.json'), 'w'))



def plot_images(images):
    """
    Plots a given set of images in a grid.

    Args:
        images (np.ndarray): The set of images to plot.

    Returns:
        None
    """
    # Create a figure and axes to plot the images
    fig, axs = plt.subplots(nrows=images.shape[0] // 4, ncols=4, figsize=(25, 25))

    # Iterate over the images and plot them on the axes
    for i in range(images.shape[0]):
        axs[i // 4, i % 4].imshow(images[i])
        axs[i // 4, i % 4].axis('off')

    # Show the plot
    plt.show()


def visualize_sample(info_file: str, n_samples: int = 8):
    """
    Visualize a sample from a custom dataset for detectron2.

    This function takes in a list of dataset records and an index for the sample to be visualized.
    It reads the image and annotations for the specified sample and displays them using detectron2's
    Visualizer class.

    Args:
        info_file (str): The name (path) of the file containing the dataset information to draw from.
        n_samples (int): The number of images to draw and view.
    """
    # retrieve the registered dataset
    dataset_dicts = json.load(open(info_file))
    dataset_name = 'visualization_only'
    register_dataset(info_file, dataset_name)

    # generate random indexes to select images
    indexes = np.random.permutation(len(dataset_dicts))

    images = []
    for i in range(n_samples):
        # Get record for specified sample
        record = dataset_dicts[indexes[i]]

        # Read image using cv2's imread function
        img = cv2.imread(record['file_name'])

        # Convert image from BGR to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create Visualizer object
        v = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(dataset_name), scale=1)

        # Use draw_dataset_dict method to draw annotations on image
        v = v.draw_dataset_dict(record)

        # Add the image to the list
        images.append(v.get_image()[:, :, ::-1])

    # Display annotated image using matplotlib imshow function
    plot_images(np.array(images))


def register_dataset(info_file, dataset_name) -> None:
    """
    registers the datasets into detectron2's acceptable format
    :param info_file: str (path-like), the json file holding the dataset information.
    :param dataset_name: str, the name of the dataset to be registered.
    :return: None
    """
    try:
        # read the file representing the current split
        dataset_dicts = json.load(open(info_file))

        # register the current data split
        DatasetCatalog.register(dataset_name, lambda: dataset_dicts)

        # Set thing_classes for the current split using MetadataCatalog.get().set()
        MetadataCatalog.get(dataset_name).set(thing_classes=['fire'])
    except AssertionError as ex:
        print(ex)


def combine(files: List[os.PathLike], splits: List[float], final_info_file: str or os.PathLike):
    """
    Combine data from multiple files into one dataset and split it into multiple parts according to the given ratios.

    :param files: A list of file paths to be combined.
    :param splits: A list of ratios for splitting the dataset.
    :param final_info_file: The file path to save the final dataset.
    """
    # Initialize an empty list to store the dataset
    dataset = []

    # Convert splits items to floats
    splits = [float(item) for item in splits]

    # Load data from each file and add it to the dataset
    for file in files:
        dataset.extend(json.load(open(file)))

    # Randomly permute the indexes of the dataset
    indexes = np.random.permutation(len(dataset))

    # Assign a unique file_id to each data point based on the permuted indexes
    for i in range(len(dataset)):
        dataset[i]['file_id'] = int(indexes[i])

    # Sort the dataset based on the file_id
    dataset = list(sorted(dataset, key=lambda x: x['file_id']))

    # Initialize an empty list to store the data splits
    data_splits = []

    # Initialize the start_index for splitting the data
    start_index = 0

    # Split the data according to the given ratios
    for split in splits:
        final_index = int(len(dataset) * split) + start_index
        data_split = dataset[start_index:final_index]
        start_index = final_index
        # Save the final dataset to a file
        json.dump(data_split, open(final_info_file + f'_split={splits.index(split)}.json', 'w'))


class DataGenerator(tf.keras.utils.Sequence, ABC):
    """A class to generate data for training or validation in batches.

    Attributes:
        images (list): A list of file paths to the images.
        masks (list): A list of file paths to the masks.
        batch_size (int): The size of the batches to generate.
        indexes (np.ndarray): An array of indexes used to shuffle the data.
    """

    def __init__(self, json_files: list, batch_size: int):
        """Initializes a DataGenerator object.

        Args:
            json_files (list): A list of file paths to the JSON files containing the data.
            batch_size (int): The size of the batches to generate.

        Returns:
            None
        """
        # Load the data from the JSON files
        data = []
        for file in json_files:
            data.extend(json.load(open(file)))

        # Extract the file paths to the images and masks
        self.images = [data_point['file_name'] for data_point in data]
        self.masks = [path.replace('image', 'mask').replace('input.jpg', 'mask.jpg') for path in self.images]

        # Set the batch size and create an array of indexes
        self.batch_size = batch_size
        self.indexes = np.arange(len(self))

    def __len__(self):
        """Calculates the number of batches per epoch.

        Returns:
            int: The number of batches per epoch.
        """
        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        """Generates a batch of data.

        Args:
            index (int): The index of the batch to generate.

        Returns:
            tuple: A tuple containing a batch of images and masks.
        """
        # Calculate the start and end indexes of the batch
        start = index * self.batch_size
        final = (index + 1) * self.batch_size

        # Load the images and masks for the batch
        images = np.array([plt.imread(image) for image in self.images[start: final]]) / 255.
        masks = np.array([plt.imread(mask) for mask in self.masks[start: final]]) / 255.

        return images, masks


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', required=True)
    parser.add_argument('--splits', nargs='+', required=True)
    parser.add_argument('--final_info_file', type=str, required=True)
    args = vars(parser.parse_args())

    # Call the combine function with the parsed arguments
    combine(**args)