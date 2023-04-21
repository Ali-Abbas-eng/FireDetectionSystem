import tarfile
import cv2
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import zipfile
import pandas as pd
import os
# import tensorflow as tf
from abc import ABC
from typing import Optional, Dict
# from detectron2.structures import BoxMode
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DistanceData(tf.keras.utils.Sequence, ABC):
    """
    DistanceData is a class that inherits from the Sequence class of Tensorflow and the ABC (Abstract Base Class).
    This class is used to extract and store data for distance estimation.

    Args:
        batch_size (int): batch size for data generation (default: 64).
        shuffle (bool): flag to shuffle the data after each epoch (default: True).
        height (int): image height (default: 1024).
        width (int): image width (default: 768).
        num_channels (int): number of image channels (default: 3).
        data_directory (str): path to the directory containing data (default: 'data/Distance Estimation data').
        val_data_separate (bool): flag to split validation data from training data (default: True).
        train_data_zip (str): path to the zip file containing training data (default: 'data/zipped/train.tar.gz').
        val_data_zip (str): path to the zip file containing validation data (default: 'data/zipped/val.tar.gz').
        split_props (float): proportion of data to use (default: 1.0).
        key_hint_pairs (Optional[Dict[str, str]]): dictionary containing keys and hints to identify data files.
        split (str): type of data split to use (default: 'train').
        unpack (bool): flag to unpack zipped data (default: False).
        create_csv (bool): flag to create csv files for training and validation data (default: False).

        This class is used to extract and store data for distance estimation. It inherits from the Sequence class of
        TensorFlow and the Abstract Base Class (ABC). The class can be used to extract data from a CSV file, as well as
        from zipped files. It provides the necessary attributes and methods to retrieve and process the data in the
        desired format. The class is highly customizable, allowing for different values to be specified for the
        batch_size, dimensions, channels, and whether the data should be shuffled. The class also provides the
        ability to create CSV files for the data.
    """

    def __init__(self,
                 batch_size: int = 64,
                 min_depth: float = .6,
                 max_depth: float = 350.,
                 shuffle: bool = True,
                 height: int = 1024,
                 width: int = 768,
                 num_channels: int = 3,
                 data_directory: str = 'data/Distance Estimation data',
                 val_data_separate: bool = True,
                 train_data_zip: str = 'data/zipped/train.tar.gz',
                 val_data_zip: str = 'data/zipped/val.tar.gz',
                 split_props: float = 1.,
                 key_hint_pairs: Optional[Dict[str, str]] = None,
                 split: str = 'train',
                 unpack: bool = False,
                 create_csv: bool = False):
        """
        Initializes the class.

        Args:
            batch_size (int): batch size for data generation (default: 64).
            shuffle (bool): flag to shuffle the data after each epoch (default: True).
            height (int): image height (default: 1024).
            width (int): image width (default: 768).
            num_channels (int): number of image channels (default: 3).
            data_directory (str): path to the directory containing data (default: 'data/Distance Estimation data').
            val_data_separate (bool): flag to split validation data from training data (default: True).
            train_data_zip (str): path to the zip file containing training data (default: 'data/zipped/train.tar.gz').
            val_data_zip (str): path to the zip file containing validation data (default: 'data/zipped/val.tar.gz').
            split_props (float): proportion of data to use (default: 1.0).
            key_hint_pairs (Optional[Dict[str, str]]): dictionary containing keys and hints to identify data files.
            split (str): type of data split to use (default: 'train').
            unpack (bool): flag to unpack zipped data (default: False).
            create_csv (bool): flag to create csv files

        Raises:
            AssertionError: if the split argument is not 'train' or 'val'

        Attributes:
        data_df (pandas DataFrame): the dataframe containing the data
        indexes (list): list of indexes corresponding to the data in the dataframe
        dim (tuple): the dimensions of the input images (height, width)
        min_depth (float): the minimum depth value
        batch_size (int): the batch size to be used during training
        shuffle (bool): whether the data should be shuffled between epochs
        n_channels (int): the number of channels in the data
        key_hint_pairs (dict): a dictionary containing the file keys and their corresponding file extensions
        split (str): the data split ('train' or 'val')
        csv_file_path (str): the path to the csv file containing the data

        Methods:
        on_epoch_end: shuffle the data indexes after each epoch
        """
        # If key_hint_pairs is not provided, set the default values for the keys
        if key_hint_pairs is None:
            key_hint_pairs = {'image': '.png', 'depth': '_depth.npy', 'mask': '_mask.npy'}

        # Ensure that the split is one of ['train', 'val']
        assert split in ['train', 'val'], f'Unknown data split value got {split}, expected one of ["train", "val"]'

        # Construct the file path for the CSV file
        csv_file_path = os.path.join(data_directory, f'data_{split}.csv')

        # If the data directory does not exist, create it
        if not os.path.isdir(data_directory):
            os.makedirs(data_directory)

        # If the create_csv flag is set, create CSV files for the data
        if create_csv:
            if val_data_separate:
                # If validation data should be separated, create separate directories for training and validation data
                train_data_directory = os.path.join(data_directory, 'train')
                val_data_directory = os.path.join(data_directory, 'val')

                # Archive the validation and training data into separate CSV files
                self.archive(val_data_directory,
                             split,
                             unpack,
                             split_props,
                             val_data_zip,
                             os.path.join(data_directory, f'data_val.csv'),
                             key_hint_pairs)
                self.archive(train_data_directory,
                             split,
                             unpack,
                             split_props,
                             train_data_zip,
                             os.path.join(data_directory, 'data_train.csv'),
                             key_hint_pairs)
            else:
                # If validation data should not be separated, use the same directory for both training and validation
                # data
                train_data_directory = data_directory
                val_data_directory = data_directory

                # Archive the training and validation data into a single CSV file
                self.archive(train_data_directory,
                             'train',
                             unpack,
                             split_props,
                             train_data_zip,
                             os.path.join(data_directory, 'data_train.csv'),
                             key_hint_pairs)
                self.archive(val_data_directory,
                             'val',
                             False,
                             split_props,
                             val_data_zip,
                             os.path.join(data_directory, f'data_val.csv'),
                             key_hint_pairs)

        self.data_df = pd.read_csv(csv_file_path)
        self.indexes = self.data_df.index.tolist()

        # Store the image dimensions (height, width) in self.dim
        self.dim = (height, width)

        # Store the minimum and maximum depth value
        self.min_depth = min_depth
        self.max_depth = max_depth

        # Store the batch size in self.batch_size
        self.batch_size = batch_size

        # Store whether the data should be shuffled or not in self.shuffle
        self.shuffle = shuffle

        # Store the number of channels in the data in self.n_channels
        self.n_channels = num_channels

        # Call the on_epoch_end method to perform any end-of-epoch operations
        self.on_epoch_end()

    @staticmethod
    def archive(data_directory, split, unpack, split_props, zipped_file_path, csv_file_path, key_hint_pairs):
        """
        Create a compressed archive of data in a given directory, split the data based on split_props, and save it to a
        CSV file.

        Args:
        - data_directory (str): the directory containing the data to be archived
        - split (str): the split of the data to be archived; must be either "train" or "val"
        - unpack (bool): whether to unpack a compressed data file
        - split_props (float): proportion of data to assign to the chosen split
        - zipped_file_path (str): path to the compressed file to unpack (if applicable)
        - csv_file_path (str): path to the CSV file where the archived data will be saved
        - key_hint_pairs (dict): a dictionary mapping data keys to file extensions

        Returns:
        - None

        """
        # create an empty dictionary to hold data corresponding to each key
        data = {key: [] for key in key_hint_pairs.keys()}

        # create an empty list to hold file paths for each file in the directory
        file_list = []

        def add_to_data(item):
            # add file path to file_list
            file_path = item
            file_list.append(file_path)

            # add file path to corresponding key in data dictionary if the file ends with an expected extension
            for key, hint in key_hint_pairs.items():
                if file_path.endswith(hint):
                    data[key].append(file_path)

        # unpack the compressed file, if requested
        if unpack:
            if zipped_file_path.endswith('.gz'):
                compressed_file = tarfile.open(name=zipped_file_path, mode='r:gz')
                members = compressed_file.getmembers()
            else:
                compressed_file = zipfile.ZipFile(zipped_file_path, 'r')
                members = compressed_file.infolist()

            # extract files from compressed file to the data directory
            for member in tqdm(members, total=len(members), desc='Unpacking'):
                compressed_file.extract(member, path=data_directory)

        # count the number of files in the data directory and create a progress bar
        num_files = sum([len(files) for _, __, files in os.walk(data_directory)])
        with tqdm(total=num_files, desc='Creating Archive') as progress_bar:
            for root, dirs, files in os.walk(data_directory):
                for file in files:
                    # add file to the data dictionary and update the progress bar
                    add_to_data(os.path.join(root, file))
                    progress_bar.update()

        # sort the data corresponding to each key in ascending order
        [data[key].sort() for key in data.keys()]

        # if split_props is specified, split the data based on the value of split_props
        if 1. > split_props > .0:
            if split == 'train':
                data = {key: data[key][:int(len(data[key]) * split_props)] for key in data.keys()}
            else:
                data = {key: data[key][int(len(data[key]) * split_props):] for key in data.keys()}

        # create a pandas DataFrame from the data dictionary and save it to a CSV file
        data_df = pd.DataFrame(data)
        data_df.to_csv(csv_file_path)

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
        return len(self.indexes) // self.batch_size

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
            x[i, ], y[i, ] = self.load(
                self.data_df['image'][batch_id],
                self.data_df['depth'][batch_id],
                self.data_df['mask'][batch_id]
            )

        # Return the loaded data as x and y
        return x, y

    def load(self,
             image_path: str,
             image_depth: str,
             image_mask: str):
        """
        Load the image, depth map and mask and process them for usage in the model.

        Parameters:
        image_path (str): str representing path to the image file
        image_depth (str): str representing path to the depth file
        image_mask (str): member representing path to the mask file

        Returns:
        Tuple of processed image and depth map as numpy arrays.
        """

        # Read the image
        image_ = plt.imread(image_path)

        # Resize the image to match the required dimensions
        image_ = cv2.resize(image_, self.dim)

        # Convert the image to a float tensor
        # image_ = tf.image.convert_image_dtype(image_, tf.float16)

        # Load the depth map and mask
        depth_map = np.load(image_depth)
        mask = np.load(image_mask)[:, :, None].astype(np.uint8)

        # Clip the depth map to exclude outliers and find the maximum depth
        depth_map = np.clip(depth_map, self.min_depth, self.max_depth)

        # Apply the mask and take the logarithm of the depth values
        depth_map *= mask

        # Resize the depth map to match the required dimensions
        depth_map = cv2.resize(depth_map, self.dim)

        # Expand the depth map to match the number of channels
        depth_map = np.expand_dims(depth_map, axis=2)

        depth_map = ((depth_map - self.min_depth) / (self.max_depth - self.min_depth)) * 2 - 1

        # Convert the depth map to a float tensor
        # depth_map = tf.image.convert_image_dtype(depth_map, tf.float16)

        return image_, depth_map

    def on_epoch_end(self):
        """
        This function updates the indexes of the data_df after each epoch.
        If the shuffle attribute is True, it shuffles the indexes.
        """
        self.indexes = np.arange(len(self.indexes))  # create an array of indices
        # shuffle the indices if shuffle attribute is True
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def plot(self, index: int = 0):
        """
        This function generates a random batch of images and depth maps and visualizes them using matplotlib.
        It creates a subplot with `batch_size` rows and 2 columns, where the first column displays the image
        and the second column displays the depth map.
        """
        images, depth_maps = self.__getitem__(
            index=index)  # get a batch of images and depth maps
        fig, ax = plt.subplots(self.batch_size, 2,
                               figsize=(50, 50))  # create a subplot with `batch_size` rows and 2 columns
        for i in range(self.batch_size):
            ax[i, 0].imshow(images[i].squeeze())  # display image
            ax[i, 1].imshow(depth_maps[i].squeeze())  # display depth map
        plt.show()  # show the plot


class FireData(DistanceData):
    def __init__(self,
                 batch_size: int = 64,
                 shuffle: bool = True,
                 height: int = 256,
                 width: int = 256,
                 num_channels: int = 3,
                 data_directory: str = os.path.join('data', 'data'),
                 train_data_zip: str = os.path.join('data', 'zipped', 'data.zip'),
                 split_props: float = .8,
                 split: str = 'train',
                 unpack: bool = False,
                 create_csv: bool = False):
        """
        Initializes a new instance of the FireData class.

        :param batch_size: int, the batch size to use during training and evaluation.
        :param shuffle: bool, whether to shuffle the data.
        :param height: int, the desired height of the images.
        :param width: int, the desired width of the images.
        :param num_channels: int, the number of channels in the images.
        :param data_directory: str, the path to the directory containing the data.
        :param train_data_zip: str, the path to the ZIP file containing the training data.
        :param split_props: float, the proportion of data to use for training.
        :param split: str, the type of data to use (train or test).
        :param unpack: bool, whether to unpack the data.
        :param create_csv: bool, whether to create a CSV file.
        """
        super().__init__(batch_size=batch_size,
                         shuffle=shuffle,
                         height=height,
                         width=width,
                         num_channels=num_channels,
                         data_directory=data_directory,
                         val_data_separate=False,
                         train_data_zip=train_data_zip,
                         val_data_zip=None,
                         split_props=split_props,
                         key_hint_pairs={'image': '_input.jpg', 'mask': '_mask.jpg'},
                         unpack=unpack,
                         split=split,
                         create_csv=create_csv)

    def __getitem__(self, index):
        """
        gets one item at the specified index
        :param index: int, location of the item to be retrieved
        :return: either a tuple that contains the image and its segmentation map, or a dictionary that contains the
                 image and its label
        """
        start = index * self.batch_size
        end = (index + 1) * self.batch_size

        # Handle edge case for end of data.
        if end >= self.data_df.shape[0]:
            end = -1

        # fetch the row of interest
        data_rows = self.data_df.loc[self.indexes[start: end]].reset_index()

        # read the image as numpy array
        images = np.array([plt.imread(data_rows['image'][ind]) for ind in range(len(data_rows))], dtype=np.float32)
        images /= 255.

        # read the mask as numpy array
        masks = np.array([plt.imread(data_rows['mask'][ind]) for ind in range(len(data_rows))], dtype=np.float32)
        masks /= 255.

        # Return the images and masks.
        return images, masks


def get_fire_dataset_dicts(file: str = os.path.join('data', 'data', 'data_train.csv')):
    data_df = pd.read_csv(file)
    for index, row in tqdm(data_df.iterrows()):
        file_name = row['image']
        image = plt.imread(file_name)
        height, width = image.shape[:2]
        mask_binary = np.where(plt.imread(row['mask']) > 0, 1, 0).astype(np.uint8)
        mask = np.where(mask_binary > 0, 255, 0).astype(np.uint8)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        record = {
            'file_name': file_name,
            'image_id': index,
            'height': height,
            'width': width,

        }
        annotations = [{'category_id': 0,
                        'bbox': [x, y, x + w, y + h],
                        'segmentation': mask,
                        'segmentation_mode': 'mask',
                        'bbox_mode': BoxMode.XYXY_ABS}
                       for x, y, w, h in [cv2.boundingRect(contour) for contour in contours]]

        record['annotations'] = annotations

