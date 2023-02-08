from abc import ABC
import os
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import shutil
from torchvision import transforms

pre_processing_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
])


class FireData(Dataset, ABC):
    @property
    def split(self):
        return self.__split

    @split.setter
    def split(self, value: str = 'train'):
        assert value in ['train', 'test', 'valid']
        self.__split = value
        self.data_df = self.splits[self.split].reset_index(drop=True)

    def __init__(self,
                 zipped_file_path: str = None,
                 data_directory: str = os.path.join('Data', 'Images'),
                 csv_file_name: str = 'data.csv',
                 train_split: float = .8,
                 valid_split: float = .1,
                 test_split: float = .1,
                 overlook_dir: str = 'segmentation',
                 split: str = 'train'):
        """
        @brief an initializer for the dataset class, handles data decompression and builds/reads the dedicated csv file
        :param zipped_file_path: str, path to the zipped file that contains the dataset, None if already decompressed
        :param data_directory: str, the path to/from which the data will be written/read
        :param csv_file_name: str, the name of the csv file that will hold the information about the dataset
        :param train_split: float, proportion of training data
        :param valid_split: float, proportion of validation data
        :param test_split: float, proportion of test data
        :param overlook_dir: str, a directory to jumpy over if not wanted in the extracted data
        """

        # create directories if they don't exist
        if not os.path.isdir(data_directory):
            os.makedirs(data_directory)

        # make sure the splits sum up to one
        assert train_split + test_split + valid_split == 1.

        # set the data directory
        self.data_dir = data_directory

        # set the directory to the images that contains fire examples
        self.fire_image_dir = os.path.join(data_directory, 'Image', 'Fire')

        # set non-fire images directory
        self.non_fire_image_dir = os.path.join(data_directory, 'Image', 'Not_Fire')

        # set segmentation images directory
        self.seg_fire_image = os.path.join(data_directory, 'segmentation')

        # set the path to the csv file to hold data info
        csv_path = os.path.join(self.data_dir, csv_file_name)

        # if the path to the zipped directory is specified
        if zipped_file_path is not None:
            # initialise labels of the images
            classes = []

            # initialise paths
            paths = []

            # initialise the names of the images to the empty list
            file_names = []

            # import zipfile for decompression functionality
            import zipfile

            # instantiate a file handle at the zipped file path
            zipped_file_object = zipfile.ZipFile(zipped_file_path)

            # for each path specified in the zip file
            for member in tqdm(zipped_file_object.namelist()):
                # if there is extra folders (or folders that is classified as unwanted in the zipped file, ignore it)
                if overlook_dir in str(member).lower():
                    continue
                _path, _file_name, _member_class = self.decompress_file(zip_file=zipped_file_object,
                                                                        data_directory=self.data_dir,
                                                                        member_object=member)
                if (_path, _file_name, _member_class) != (None, None, None):
                    # add the file to the list
                    paths.append(_path)

                    # append the file name to the list of file names (to be added to the data frame)
                    file_names.append(_file_name)

                    # append the class of the current file to the list of classes (to be added to the data frame)
                    classes.append(_member_class)

            # instantiate a dataframe object with previously obtained information
            data_df = pd.DataFrame({'dir': paths,
                                    'file': file_names,
                                    'class': classes,})
            if not os.path.isfile(csv_path):
                # save the extracted data to a csv file
                data_df.to_csv(csv_path, index=False)

            else:
                temp = pd.concat([pd.read_csv(csv_path), data_df])
                temp.to_csv(csv_path, index=False)

        # read the data (hopefully store correctly in a csv file)
        self.data_df = pd.read_csv(csv_path)

        self.train_split = train_split
        self.valid_split = valid_split
        self.test_split = test_split

        # aggregate everything in one dictionary
        self.splits = self.split_data()

        # make sure the user input a valid name for the split to be used
        assert split in self.splits.keys()

        # save the split value
        self.split = split

        # get the corresponding data
        self.data_df = self.splits[self.split]

    def split_data(self):
        # specify splits of data
        # split the data into positive and negative examples
        positive_examples = self.data_df.loc[self.data_df['class'] == 'fire']
        negative_examples = self.data_df.loc[self.data_df['class'] == 'non-fire']

        # number of examples in the training split

        # get the number of training positive examples
        train_end_positive = int(positive_examples.shape[0] * self.train_split)

        # training negative examples
        train_end_negative = int(negative_examples.shape[0] * self.train_split)

        # find at which index the validation ends
        valid_end_positive = train_end_positive + int(positive_examples.shape[0] * self.valid_split)
        valid_end_negative = train_end_negative + int(negative_examples.shape[0] * self.valid_split)

        # build the training data
        train_data_df = pd.concat([
            positive_examples.loc[:train_end_positive],
            negative_examples.loc[:train_end_negative]
        ]).reset_index(drop=True)

        # build the validation data
        valid_data_df = pd.concat([
            positive_examples[train_end_positive:valid_end_positive],
            negative_examples[train_end_negative:valid_end_negative]
        ]).reset_index(drop=True)

        # build the test data
        test_data_df = pd.concat([
            positive_examples[valid_end_positive:],
            negative_examples[valid_end_negative:]
        ]).reset_index(drop=True)

        return {
            'train': train_data_df,
            'test': test_data_df,
            'valid': valid_data_df
        }

    def decompress_file(self, zip_file, member_object, data_directory):
        file_name = os.path.basename(member_object)  # specific name of the file

        if not file_name:
            return None, None, None

        # specify the class of the current image
        member_class = self.get_class(str(member_object))

        # set the path to the file to be extracted
        path_to_buffered_file = os.path.join(data_directory, member_class)

        # if the target directory doesn't exist
        if not os.path.isdir(path_to_buffered_file):
            # create it
            os.makedirs(path_to_buffered_file)

        path_to_buffered_file = os.path.join(path_to_buffered_file, file_name)  # add file name to path

        # extract -overlook the type (file or directory)-
        source = zip_file.open(member_object)  # specify the file to be extracted
        target = open(path_to_buffered_file, 'wb')  # specify the binary placeholder
        with source, target:
            shutil.copyfileobj(source, target)  # write the zipped file into the binary placeholder

        return os.path.join(data_directory, member_class), file_name, member_class

    @staticmethod
    def get_class(path):
        if '/fire' in path.lower():
            class_name = 'fire'
        else:
            class_name = 'non-fire'
        return class_name

    def __len__(self):
        """
        returns number of rows in the final data_df
        :return: int, length of the dataset
        """
        return self.data_df.shape[0]

    def __getitem__(self, index: int) -> list:
        """
        gets one item at the specified index
        :param index: int, location of the item to be retrieved
        :return: either a tuple that contains the image and its segmentation map, or a dictionary that contains the
         image and its label
        """
        # fetch the row of interest
        data_row = self.data_df.loc[index]

        # specify the type of the image
        image_class = data_row['class']

        # specify directory to said image
        image_path = os.path.join(data_row['dir'], data_row['file'])

        # read the image as numpy array
        image = Image.open(image_path)

        # return the image whither or not it holds the fire class (1 if class == fire)
        return [image, int(image_class == 'fire')]


class FireDataSegmentation(FireData):
    def __init__(self,
                 zipped_file_path: str = None,
                 data_directory: str = os.path.join('Data', 'Images (segmentation)'),
                 csv_file_name: str = 'data.csv',
                 train_split: float = .8,
                 valid_split: float = .1,
                 test_split: float = .1,
                 overlook_dir: str = 'not_fire',
                 split: str = 'train'):
        super().__init__(zipped_file_path,
                         data_directory,
                         csv_file_name,
                         train_split,
                         valid_split,
                         test_split,
                         overlook_dir,
                         split)

    @staticmethod
    def get_class(path):
        if 'segmentation' in path.lower():
            class_name = 'segmentation'
        elif 'fire' in path.lower():
            class_name = 'fire'
        else:
            class_name = 'non-fire'
        return class_name

    @staticmethod
    def pre_process(image):
        return pre_processing_pipeline(image)

    def __getitem__(self, index) -> list:
        """
        gets one item at the specified index
        :param index: int, location of the item to be retrieved
        :return: either a tuple that contains the image and its segmentation map, or a dictionary that contains the
         image and its label
        """
        # fetch the row of interest
        data_row = self.data_df.loc[index]

        # specify directory to said image
        image_path = os.path.join(data_row['dir'], data_row['file'])

        # read the image as numpy array
        image = Image.open(image_path)

        image = self.pre_process(image)

        # get the path to the corresponding segmentation map
        seg_path = os.path.join(self.seg_fire_image, data_row['file'])

        # read the image at the specified path
        seg_image = Image.open(seg_path)

        seg_image = self.pre_process(seg_image)

        # return image and corresponding segmentation
        return [image, seg_image]


if __name__ == '__main__':
    FireDataSegmentation()