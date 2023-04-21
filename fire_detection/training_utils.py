"""
This file contains two custom callback classes for use during training of a Keras model.

The `ProgressCallback` class displays the progress of training and testing using a progress bar. It has attributes for
the total number of epochs, the total number of batches in the training and test data, and the progress bar object.

The `LRDecayCallback` class implements learning rate decay during training. It has attributes for the decay factor,
the number of epochs to wait before decaying the learning rate, the minimum learning rate value,
the history of training losses, the best training loss achieved so far, and a counter to keep track of the number of
epochs since the last improvement in training loss.
"""

import tensorflow as tf
from tqdm.auto import tqdm


class ProgressCallback(tf.keras.callbacks.Callback):
    """
    A custom callback class to display the progress of training and testing using a progress bar.

    Attributes:
        epochs (int): The total number of epochs for training.
        train_total (int): The total number of batches in the training data.
        test_total (int): The total number of batches in the test data.
        progress_bar (tqdm): The progress bar object used to display the progress.
    """

    def __init__(self, epochs, train_total, test_total):
        """
        Initializes a ProgressCallback object.

        Args:
            epochs (int): The total number of epochs for training.
            train_total (int): The total number of batches in the training data.
            test_total (int): The total number of batches in the test data.

        Returns:
            None
        """
        super().__init__()
        self.epochs = epochs
        self.train_total = train_total
        self.test_total = test_total
        self.progress_bar = None

    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the beginning of each epoch.

        Args:
            epoch (int): The current epoch number.
            logs (dict): A dictionary containing the logs from the previous epoch.

        Returns:
            None
        """
        if logs:
            self.progress_bar.set_postfix(logs)

        # Create a new progress bar for the current epoch
        self.progress_bar = tqdm(total=self.train_total, unit=' batch', desc=f'Epoch {epoch:04}'.rjust(15))

    def on_train_batch_end(self, batch, logs=None):
        """
        Called at the end of each training batch.

        Args:
            batch (int): The current batch number.
            logs (dict): A dictionary containing the logs from the current batch.

        Returns:
            None
        """
        if logs:
            self.progress_bar.set_postfix(logs)

        # Update the progress bar with the current batch
        self.progress_bar.update()

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            logs (dict): A dictionary containing the logs from the current epoch.

        Returns:
            None
        """
        if logs:
            self.progress_bar.set_postfix(logs)

        # Close the progress bar for the current epoch
        self.progress_bar.close()

    def on_test_begin(self, logs=None):
        """
        Called at the beginning of testing.

        Args:
            logs (dict): A dictionary containing the logs from training.

        Returns:
            None
        """

        # Create a new progress bar for testing
        self.progress_bar = tqdm(total=self.test_total, unit=' batch', desc=f'{"Testing".rjust(15)}')

    def on_test_batch_end(self, batch, logs=None):
        """
        Called at the end of each test batch.

        Args:
            batch (int): The current batch number.
            logs (dict): A dictionary containing the logs from the current batch.

        Returns:
            None
        """
        if logs:
            self.progress_bar.set_postfix(logs)

        # Update the progress bar with the current batch
        self.progress_bar.update()


class LRDecayCallback(tf.keras.callbacks.Callback):
    """
    A custom callback class to implement learning rate decay during training.

    Attributes:
        factor (float): The factor by which to decay the learning rate.
        patience (int): The number of epochs to wait before decaying the learning rate.
        min_lr (float): The minimum learning rate value.
        loss_history (list): A list containing the history of training losses.
        best_loss (float): The best training loss achieved so far.
        counter (int): A counter to keep track of the number of epochs since the last improvement in training loss.
    """

    def __init__(self, learning_rate: float = .001, factor=0.1, patience=10, min_lr=1e-6):
        """
        Initializes a LRDecayCallback object.

        Args:
            learning_rate (float): The initial learning rate.
            factor (float): The factor by which to decay the learning rate.
            patience (int): The number of epochs to wait before decaying the learning rate.
            min_lr (float): The minimum learning rate value.

        Returns:
            None
        """
        super(LRDecayCallback, self).__init__()
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.loss_history = []
        self.best_loss = float('inf')
        self.counter = 0

    def on_train_batch_end(self, batch, logs=None):
        """
        Called at the end of each training batch.

        Args:
            batch (int): The current batch number.
            logs (dict): A dictionary containing the logs from the current batch.

        Returns:
            None
        """

        # Get the current training loss from the logs
        current_loss = logs.get('loss')
        self.loss_history.append(current_loss)

        # Check if the current loss is better than the best loss so far
        if current_loss < self.best_loss:
            # Update the best loss and reset the counter
            self.best_loss = current_loss
            self.counter = 0
        else:
            # Increment the counter
            self.counter += 1

            # Check if the counter has reached the patience value
            if self.counter >= self.patience:
                # Decay the learning rate by multiplying it with the decay factor
                lr = tf.keras.backend.get_value(self.model.optimizer.lr)
                new_lr = max(lr * self.factor, self.min_lr)
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)

                # Reset the counter
                self.counter = 0

