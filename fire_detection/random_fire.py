import argparse
import os
from data_tools import create_record, visualize_sample, extract
import matplotlib.pyplot as plt


def main(zip_file: str or os.PathLike,
         output_dir: str or os.PathLike):
    """
    Extracts data from a zip file.

    Args:
        zip_file (str or os.PathLike): The path to the zip file containing the data.
        output_dir (str or os.PathLike): The path to the directory where the extracted data will be saved.

    Returns:
        None
    """
    # Extract the data from the zip file
    extract(zip_file=zip_file,
            output_dir=output_dir,
            image_mark='input.jpg',
            mask_mark='mask.jpg',
            pre_process=lambda x: None,
            output_info_file='random_fire',
            mask_processor=lambda x: plt.imread(x),
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
    visualize_sample(os.path.join(args['output_dir'], 'random_fire_test.json'))