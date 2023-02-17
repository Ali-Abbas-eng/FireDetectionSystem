import os
import tarfile
from abc import ABC
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import os
import tarfile
from abc import ABC
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import zipfile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
                 batch_size: int = 64,
                 shuffle: bool = True,
                 height: int = 1024,
                 width: int = 768,
                 num_channels: int = 3,
                 data_directory: str = os.path.join('Data', 'Distance Estimation Data'),
                 val_data_separate: bool = True,
                 train_data_zip: str = os.path.join('Data', 'zipped', 'train.tar.gz'),
                 val_data_zip: str = os.path.join('Data', 'zipped', 'val.tar.gz'),
                 split_props: float = 1.,
                 key_hint_pairs=None,
                 split: str = 'train',
                 unpack: bool = False,
                 create_csv: bool = False):

        if key_hint_pairs is None:
            key_hint_pairs = {'image': '.png', 'depth': '_depth.npy', 'mask': '_mask.npy'}
        assert split in ['train', 'val'], f'Unknown data split value got {split}, expected one of ["train", "val"]'
        csv_file_path = os.path.join(data_directory, f'data_{split}.csv')
        if not os.path.isdir(data_directory):
            os.makedirs(data_directory)
        if create_csv:
            if val_data_separate:
                train_data_directory = os.path.join(data_directory, 'train')
                val_data_directory = os.path.join(data_directory, 'val')
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
                train_data_directory = data_directory
                val_data_directory = data_directory
                self.archive(train_data_directory,
                             split,
                             unpack,
                             split_props,
                             train_data_zip,
                             os.path.join(data_directory, 'data_train.csv'),
                             key_hint_pairs)
                self.archive(val_data_directory,
                             split,
                             False,
                             split_props,
                             val_data_zip,
                             os.path.join(data_directory, f'data_val.csv'),
                             key_hint_pairs)


        self.data_df = pd.read_csv(csv_file_path)
        self.indexes = self.data_df.index.tolist()

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

    @staticmethod
    def archive(data_directory,
                split,
                unpack,
                split_props,
                zipped_file_path,
                csv_file_path,
                key_hint_pairs):
        data = {key: [] for key in key_hint_pairs.keys()}
        file_list = []

        def add_to_data(item):
            file_path = item
            file_list.append(file_path)
            for key, hint in key_hint_pairs.items():
                if file_path.endswith(hint):
                    data[key].append(file_path)


        if unpack:
            if zipped_file_path.endswith('.gz'):
                compressed_file = tarfile.open(name=zipped_file_path, mode='r:gz')
                members = compressed_file.getmembers()
            else:
                compressed_file = zipfile.ZipFile(zipped_file_path, 'r')
                members = compressed_file.infolist()

            for member in tqdm(members, total=len(members), desc='Unpacking'):
                compressed_file.extract(member, path=data_directory)

        num_files = sum([len(files) for _, __, files in os.walk(data_directory)])
        with tqdm(total=num_files, desc='Creating Archive') as progress_bar:
            for root, dirs, files in os.walk(data_directory):
                for file in files:
                    add_to_data(os.path.join(root, file))
                    progress_bar.update()

        [data[key].sort() for key in data.keys()]
        if 1. > split_props > .0:
            if split == 'train':
                data = {key: data[key][:int(len(data[key]) * split_props)] for key in data.keys()}
            else:
                data = {key: data[key][int(len(data[key]) * split_props):] for key in data.keys()}

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
        image_path (TarInfo): member representing the image file
        image_depth (TarInfo): member representing the depth file
        image_mask (TarInfo): member representing the mask file

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
        mask = np.load(image_mask)[:, :, None]
        mask = mask > 0

        # Clip the depth map to exclude outliers and find the maximum depth
        max_depth = min(300, max(np.percentile(depth_map, 99), .001))
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
                 data_directory: str = os.path.join('Data', 'Fire Data'),
                 train_data_zip: str = os.path.join('Data', 'zipped', 'Fire Data.zip'),
                 split_props: float = .8,
                 split: str = 'train',
                 unpack: bool = False,
                 create_csv: bool = False):
        super().__init__(batch_size,
                         shuffle,
                         height,
                         width,
                         num_channels,
                         data_directory,
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

        if end >= self.data_df.shape[0]:
            end = -1

        # fetch the row of interest
        data_rows = self.data_df.loc[self.indexes[start: end]].reset_index()

        # read the image as numpy array
        images = np.array([plt.imread(data_rows['image'][ind]) for ind in range(len(data_rows))], dtype=np.float32)
        images /= 255.

        masks = np.array([plt.imread(data_rows['mask'][ind]) for ind in range(len(data_rows))], dtype=np.float32)
        masks /= 255.

        # return the image whither or not it holds the fire class (1 if class == fire)
        return images, masks
