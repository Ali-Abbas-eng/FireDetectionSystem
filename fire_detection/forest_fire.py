import argparse
import json
import os
import random
from zipfile import ZipFile
from PIL import Image
from data_tools import create_record, visualize_sample, extract
import numpy as np
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt


def mask_handler(mask_file_name):
    """
    Processes a mask image file and returns a binary mask.

    Args:
        mask_file_name (str): The name of the mask image file.

    Returns:
        np.ndarray: A binary mask where 1 indicates the presence of the maximum value in the original mask.
    """
    # Open the mask image file
    mask_image = Image.open(mask_file_name)

    # Convert the image to a numpy array
    mask_image = np.array(mask_image)

    # Calculate the mean of the RGB values along the third axis
    mask_image = np.mean(mask_image, axis=2)

    # Find the unique values in the mask
    unique_values = np.unique(mask_image)

    # Find the maximum value in the mask
    maximum = max(unique_values)

    # Create a binary mask where 1 indicates the presence of the maximum value in the original mask
    mask_image = np.where(mask_image == maximum, 1, 0)

    # Return the binary mask
    return mask_image


def image_pre_process(file):
    """
    Pre-processes an image by resizing it and saving it to the same file.

    Args:
        file (str): The name of the image file.

    Returns:
        None
    """
    # Open the image file
    image = Image.open(file)

    # Convert the image to a numpy array
    image = np.array(image)

    # Resize the image to 1/8th of its original size
    image = cv2.resize(image, (image.shape[1] // 8, image.shape[0] // 8))

    # Save the pre-processed image to the same file
    plt.imsave(file, image)


def main(zip_file: str or os.PathLike,
         output_dir: str or os.PathLike):
    """
    Extracts data from a zip file and pre-processes it.

    Args:
        zip_file (str or os.PathLike): The path to the zip file containing the data.
        output_dir (str or os.PathLike): The path to the directory where the extracted data will be saved.

    Returns:
        None
    """
    # Extract the data from the zip file and pre-process it
    extract(zip_file=zip_file,
            output_dir=output_dir,
            image_mark='input.png',
            mask_mark='mask.png',
            pre_process=image_pre_process,
            output_info_file='forest_fire',
            mask_processor=mask_handler,
            train_split=.8)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = vars(parser.parse_args())

    # Call the main function with the parsed arguments
    main(**args)

    # Visualize a sample from the test set
    visualize_sample(os.path.join(args['output_dir'], 'forest_fire_test.json'))


