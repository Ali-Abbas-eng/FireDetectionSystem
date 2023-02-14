import os
import tarfile
from abc import ABC
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.backend.set_floatx('float16')


class DistanceData(tf.keras.utils.Sequence, ABC):
    """
    DistanceData is a class that inherits from the Sequence class of Tensorflow and the ABC (Abstract Base Class)
    This class is used to extract and store data for distance estimation

    Args:
    data_csv_file (str): path to the data.csv file
    data_directory (str): path to the data directory
    unpack (bool): flag to unpack the zipped data
    zipped_data_file_path (str): path to the zipped data file
    """

    def __init__(self,
                 data_csv_file: str = os.path.join('Data', 'Distance Estimation Data', 'data.csv'),
                 batch_size: int = 64,
                 shuffle: bool = True,
                 height: int = 768,
                 width: int = 1024,
                 num_channels: int = 3,
                 data_directory: str = os.path.join('Data', 'Distance Estimation Data'),
                 unpack: bool = False,
                 split: str = 'train',
                 train_proportion: float = .1,
                 zipped_data_file_path: str = os.path.join('Data', 'zipped', 'val.tar.gz')):

        # If unpack flag is True, extract the zipped data to data_directory
        if unpack:
            # Create the data directory if it does not exist
            if not os.path.isdir(data_directory):
                os.makedirs(data_directory)
            # Check if zipped data file exists
            assert os.path.isfile(zipped_data_file_path)
            # Open the zipped data file
            tar_file = tarfile.open(zipped_data_file_path)

            # get the members of the tar.gz file
            members = tar_file.getmembers()

            # Get the list of files after extraction
            file_list = []

            with tqdm(total=len(members), desc='Extracting Files') as progress_bar:
                # Loop through each member in the tar file
                for member in members:
                    if len(member.name.split('/')) < 2:
                        progress_bar.update()
                        continue
                    # Get the file path of the extracted member
                    member_path = os.path.join(*[data_directory, *member.name.split('/')])

                    # Extract the current member (file or directory)
                    tar_file.extract(member, path=data_directory)

                    # Check if the extracted member is a file
                    if os.path.isfile(member_path):
                        # If it is a file, append it to the list of extracted files
                        file_list.append(member_path)

                    # Update the progress bar
                    progress_bar.update()

            # Close the tar file
            tar_file.close()

            # Sort the file list
            file_list.sort()

            # Create a dictionary of image, depth, and mask files
            data = {
                'image': [x for x in file_list if x.endswith('.png')],
                'depth': [x for x in file_list if x.endswith('_depth.npy')],
                'mask': [x for x in file_list if x.endswith('_depth_mask.npy')],
            }
            # Add the class label ('indoors' or 'outdoor') to the dictionary based on the file name
            data['class'] = ['indoors' if 'indoors' in x else 'outdoor' for x in data['image']]

            # Create a data frame from the dictionary
            df = pd.DataFrame(data)
            # Save the data frame to csv file
            df.to_csv(data_csv_file, index=False)

        # Check if data_csv_file exists
        assert os.path.isfile(data_csv_file)

        # Load the data from the CSV file and store it in self.data_df
        self.data_df = pd.read_csv(data_csv_file)

        # Store the index values of the data as a list in self.indexes
        self.indexes = self.data_df.index.tolist()
        if split == 'train':
            self.indexes = self.indexes[:int(len(self.indexes) * train_proportion)]
        elif split == 'valid':
            self.indexes = self.indexes[:int(len(self.indexes) * (1 - train_proportion))]

        # Store the image dimensions (height, width) in self.dim
        self.dim = (height, width)

        # Store the minimum depth value in self.min_depth
        self.min_depth = .1

        # Store the batch size in self.batch_size
        self.batch_size = batch_size

        # Store whether the data should be shuffled or not in self.shuffle
        self.shuffle = shuffle

        # Store the number of channels in the data in self.n_channels
        self.n_channels = num_channels

        # Call the on_epoch_end method to perform any end-of-epoch operations
        self.on_epoch_end()

    # Define the method to return the number of batches in the dataset
    def __len__(self):
        """
        This method returns the number of batches in the dataset. The number of batches is calculated
        by dividing the total number of data samples by the batch size and rounding up to the nearest
        integer value.

        Returns:
            int: the number of batches in the dataset
        """
        # Return the number of batches in the dataset by dividing the number of data samples by the batch size and
        # rounding up
        return int(np.ceil(len(self.indexes) / self.batch_size))

    # Define the method to retrieve a batch of data from the dataset
    def __getitem__(self, index):
        """
        This method retrieves a batch of data from the dataset. The size of the batch may be less than
        the specified batch size if the last batch in the dataset is smaller. The data is generated
        by calling the data_generation method.

        Args:
            index (int): the index of the batch to retrieve

        Returns:
            tuple: a tuple of the inputs and targets for the batch
        """
        # set the index of the first element in the batch to be returned
        start = index * self.batch_size

        # set the index of the last element in the batch to be returned
        end = (index + 1) * self.batch_size

        # handle the special case of the last batch
        if end > len(self.indexes):
            end = -1

        # Get the index values for the current batch
        index = self.indexes[start: end]
        batch = [self.indexes[k] for k in index]

        # Generate the inputs and targets for the batch
        x, y = self.data_generation(batch)

        # Return the inputs and targets for the batch
        return x, y

    def data_generation(self, batch):
        """
        This function generates the input and target data for the model, given a batch of indices.
        The function loads the corresponding image, depth, and mask data for each index in the batch,
        and returns the loaded data as x and y respectively.

        :param batch: A list of indices to be used to retrieve the image, depth, and mask data
        :return: x, y; input and target data respectively.
        """
        # Initialize x with an empty array of size (batch_size, height, width, num_channels)
        x = np.empty((self.batch_size, *self.dim[::-1], self.n_channels))
        # Initialize y with an empty array of size (batch_size, height, width, 1)
        y = np.empty((self.batch_size, *self.dim[::-1], 1))

        # Loop through each index in the batch
        for i, batch_id in enumerate(batch):
            # Load the image, depth, and mask data for the current index
            x[i,], y[i,] = self.load(
                self.data_df['image'][batch_id],
                self.data_df['depth'][batch_id],
                self.data_df['mask'][batch_id]
            )

        # Return the loaded data as x and y
        return x, y

    def load(self, image_path: str, image_depth: str, image_mask: str):
        """
        Load the image, depth map and mask and process them for usage in the model.

        Parameters:
        image_path (str): path to the image file
        image_depth (str): path to the depth map file
        image_mask (str): path to the mask file

        Returns:
        Tuple of processed image and depth map as numpy arrays.
        """

        # Read the image
        image_ = plt.imread(image_path)

        # Resize the image to match the required dimensions
        image_ = cv2.resize(image_, self.dim)

        # Convert the image to a float tensor
        image_ = tf.image.convert_image_dtype(image_, tf.float16)

        # Load the depth map and mask
        depth_map = np.load(image_depth)
        mask = np.load(image_mask)[:, :, None]
        mask = mask > 0

        # Clip the depth map to exclude outliers and find the maximum depth
        max_depth = min(300, np.percentile(depth_map, 99))
        depth_map = np.clip(depth_map, self.min_depth, max_depth)

        # Apply the mask and take the logarithm of the depth values
        depth_map = np.log(depth_map, where=mask)
        depth_map = np.ma.masked_where(~mask, depth_map)

        # Clip the logarithm of the depth values
        depth_map = np.clip(depth_map, .1, np.log(max_depth))

        # Resize the depth map to match the required dimensions
        depth_map = cv2.resize(depth_map, self.dim)

        # Expand the depth map to match the number of channels
        depth_map = np.expand_dims(depth_map, axis=2)

        # Convert the depth map to a float tensor
        depth_map = tf.image.convert_image_dtype(depth_map, tf.float16)

        return image_, depth_map

    def on_epoch_end(self):
        """
        This function updates the indexes of the data_df after each epoch.
        If the shuffle attribute is True, it shuffles the indexes.
        """
        self.indexes = np.arange(len(self.indexes))  # create an array of indices
        if self.shuffle:  # shuffle the indices if shuffle attribute is True
            np.random.shuffle(self.indexes)

    def visualise_depth_map(self):
        """
        This function generates a random batch of images and depth maps and visualizes them using matplotlib.
        It creates a subplot with `batch_size` rows and 2 columns, where the first column displays the image
        and the second column displays the depth map.
        """
        images, depth_maps = self.__getitem__(
            index=np.random.randint(len(self)))  # get a random batch of images and depth maps
        fig, ax = plt.subplots(self.batch_size, 2,
                               figsize=(50, 50))  # create a subplot with `batch_size` rows and 2 columns
        for i in range(self.batch_size):
            ax[i, 0].imshow(images[i].squeeze())  # display image
            ax[i, 1].imshow(depth_maps[i].squeeze())  # display depth map
        plt.show()  # show the plot


if __name__ == '__main__':
    data_generator = DistanceData(batch_size=4)
    from tqdm import tqdm

    with tqdm(total=len(data_generator), desc='Iterating through Data') as progress_bar:
        mean = .0
        std = .0
        maximum = -np.inf
        minimum = np.inf
        for i in range(len(data_generator)):
            x_, y_ = data_generator[i]
            mean += np.mean(y_) / (i + 1)
            std += np.std(y_) / (i + 1)
            maximum = max(maximum, np.max(y_))
            minimum = min(minimum, np.min(y_))
            progress_bar.set_postfix_str(s=f'Mean: {mean} | St dev.: {std} | Max: {maximum} | Min: {minimum}')
            progress_bar.update()

    data_generator.visualise_depth_map()
