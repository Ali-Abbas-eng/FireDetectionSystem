"""
This file contains two functions for training a model for image segmentation using TensorFlow.

The `train` function trains a given model using the provided training and test data generators,
learning rate, decay factor, and number of epochs. It takes as input the model to be trained,
 the generators for the training and test data, the initial learning rate to use during training,
  the factor by which to decay the learning rate, and the number of passes through the dataset.

The `main` function encapsulates the training process. It takes as input a list of paths to json files representing the
train and test sets, the batch size to use during training, the initial learning rate to use during training,
 the factor by which to decay the learning rate, and the number of passes through the dataset.
 It creates data generators for the train and test sets and calls the `train` function to train the model.
"""

import argparse
from data_tools import DataGenerator
from network import get_fire_segmentation_model
import os
import tensorflow as tf
import tensorflow_addons as tfa


def train(model, train_generator, test_generator, learning_rate, decay_factor, epochs):
    """
    Trains a model for image segmentation using TensorFlow.

    :param model: The model to be trained.
    :param train_generator: The generator for the training data.
    :param test_generator: The generator for the test data.
    :param learning_rate: The initial learning rate to use during training.
    :param decay_factor: The factor by which to decay the learning rate.
    :param epochs: Number of passes on the Dataset
    """
    # Create a callback to decay the learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=decay_factor
    )

    # Create a TensorBoard callback to log the training information
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join('fire_detection', 'models', 'logs'))

    # Create model checkpointing callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('fire_detection',
                                                                                   'models',
                                                                                   'model.h5'),
                                                             load_weights_on_restart=True,
                                                             monitor='val_loss',
                                                             save_weights_only=True,
                                                             save_best_only=True,
                                                             mode='min',
                                                             save_freq='epoch')

    # Compile the model with the specified learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=[tfa.metrics.F1Score(num_classes=2),
                 tf.keras.metrics.Precision(),
                 tf.keras.metrics.AUC(),
                 tf.keras.metrics.Accuracy()]
    )

    # Train the model
    model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=epochs,
        callbacks=[tensorboard_callback, checkpoint_callback]
    )


def main(train_files, test_files, batch_size, learning_rate, decay_factor, epochs):
    """
    Encapsulates the training process.

    Args:
        train_files (list): A list of paths to json files each containing a list of dictionaries representing the train set.
        test_files (list): A list of paths to json files each containing a list of dictionaries representing the test set.
        batch_size (int): The number of images per training step.
        learning_rate (int): The alpha parameter to control the speed of learning.
        decay_factor (float): A factor with which the learning rate will be multiplied during training.
        epochs (int): The number of passes through the dataset.

    Returns:
        None
    """
    # Create data generators for the train and test sets
    train_generator = DataGenerator(json_files=train_files, batch_size=batch_size)
    test_generator = DataGenerator(json_files=test_files, batch_size=batch_size)

    # Get the fire segmentation model
    model = get_fire_segmentation_model()

    # Train the model
    train(model, train_generator, test_generator, learning_rate, decay_factor, epochs)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_files', nargs='+', default=[os.path.join('fire_detection',
                                                                          'data',
                                                                          'random_fire',
                                                                          'random_fire_train.json')])
    parser.add_argument('--test_files', nargs='+', default=[os.path.join('fire_detection',
                                                                         'data',
                                                                         'random_fire',
                                                                         'random_fire_test.json')])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=.001)
    parser.add_argument('--decay_factor', type=float, default=.9)
    args = vars(parser.parse_args())
    main(**args)
