import re
from torch.nn import (
    ConvTranspose2d,
    Conv2d,
    MaxPool2d,
    Module,
    ModuleList,
    ReLU,
    functional,
    Sigmoid,
)
from pytorch_toolbelt.losses import JaccardLoss
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from typing import List, Tuple, Callable, Any
from tqdm import tqdm
from fire_data_prep import FireDataSegmentation
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from torchmetrics import Accuracy, AUROC, Precision, Recall, F1Score, JaccardIndex, Dice
import torch


class Block(Module):
    """
    defines one block of convolutional layers of the UNet model
    """

    def __init__(self, input_channels: int, output_channels: int) -> None:
        """
        initialises the block with the number of channels in the input and the desired number of output channels
        :param input_channels: int, the value of the channels dimension
        :param output_channels: int, the desired value of the output channels
        """
        # call to the super class initializer
        super().__init__()

        # the first convolutional layer in the block
        self.conv1 = Conv2d(input_channels, output_channels, 3)

        # activation
        self.relu = ReLU()

        # second convolution
        self.conv2 = Conv2d(output_channels, output_channels, 3)

    def forward(self, x) -> torch.Tensor:
        # propagate through the block
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(Module):
    """
    the encoder of the UNet model (to reach the bottleneck)
    """

    def __init__(self, channels: Tuple[int] = (3, 16, 32, 64)) -> None:
        """
        initializer of the encoder
        :param channels: desired channels in the blocks
        """
        # call to the super
        super().__init__()

        # stack of layers in the encoder
        blocks = [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]

        # encapsulation of the blocks to be forward propagate-able
        self.encoder_blocks = ModuleList(blocks)

        # max pooling the select candidate features
        self.pool = MaxPool2d(2)

    def forward(self, x) -> List[torch.Tensor]:
        # initialise an empty list to hold the outputs
        block_outputs = []

        # loop through the network's blocks
        for block in self.encoder_blocks:
            # propagate through current layer
            x = block(x)

            # append the output to the list of outputs
            block_outputs.append(x)

            # select candidate features
            x = self.pool(x)

        # return the output of each block
        return block_outputs


class Decoder(Module):
    def __init__(self, channels: Tuple[int] = (64, 32, 16)) -> None:
        """
        re-build the data from the bottleneck up to a one-channel segmentation map
        :param channels: tuple, number of channels in the consecutive blocks of the decoder
        """
        # call to the super class' initializer
        super().__init__()

        # save the number of channels to the model
        self.channels = channels

        # specify the list of decoder blocks
        self.up_convs = ModuleList([
            ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)
        ])

        # set the decoder blocks
        self.decoder_blocks = ModuleList([
            Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)
        ])

    def forward(self, x: torch.Tensor, encoder_features: torch.Tensor) -> torch.Tensor:
        """
        forward propagation through the decoder
        :param x: torch.Tensor, the final output of the encoder
        :param encoder_features: torch.Tensor, the output of the individual layers of the encoder
        :return:
        """
        # loop through the decoder blocks
        for i in range(len(self.channels) - 1):
            # propagate x through the appropriate ConvTranspose layer
            x = self.up_convs[i](x)

            # crop the output of the corresponding block in the encoder to match the shape of the input x
            encoder_feature = self.crop(encoder_features[i], x)

            # concatenate the encoder's output with the ConvTranspose output
            x = torch.cat([x, encoder_feature], dim=1)

            # propagate the concatenated outputs through the relevant decoder block
            x = self.decoder_blocks[i](x)

        # return the result of the final block of convolutions
        return x

    @staticmethod
    def crop(encoder_features, x) -> torch.Tensor:
        """
        a helper function to get the encoder features to match the shape of the input to the decoder
        :param encoder_features: torch.Tensor, the output of one encoder's block
        :param x: torch.Tensor, the output of some level of the decoder
        :return:
        """
        # extract the height and width of the produced output
        _, _, h, w = x.shape

        # crop a box from the output that preserves the center
        encoder_features = transforms.CenterCrop([h, w])(encoder_features)
        return encoder_features


class UNet(Module):
    def __init__(self,
                 encoder_channels=(3, 16, 32, 64),
                 decoder_channels=(64, 32, 16),
                 num_classes=1,
                 retrain_dim=True,
                 output_size=(256, 256),
                 model_directory: str = None,
                 checkpoint_index: int = None,
                 base_name: str = 'UNet') -> None:
        """
        encapsulation of the UNet model
        :param encoder_channels: tuple, input channels in the decoder (always starts with 3)
        :param decoder_channels: tuple, output channels of the decoder
        :param num_classes: int, number of classes for which we need to extract segmentation
        :param retrain_dim: bool, make sure the model outputs the same shape (height and width) of the input
        :param output_size: tuple, height and width of the output
        :param model_directory: str, the directory which holds the saved checkpoints of the model
        """
        # invoking the initializer of the super class
        super().__init__()
        # set the encoder block
        self.encoder = Encoder(encoder_channels)

        # define the decoder block
        self.decoder = Decoder(decoder_channels)

        # set the device to work on
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # define the last layer in the model so that it produces the same image-size as the masks
        self.head_conv = Conv2d(decoder_channels[-1], num_classes, 1)

        # define the activation function of the last layer to classify pixels
        self.head_activation = Sigmoid()

        # set the retain_dim attribute
        self.retrain_dim = retrain_dim

        # set the output_size property
        self.output_size = output_size

        # set number of classes to handle
        self.num_classes = num_classes
        # check the existence of the models directory
        if not os.path.isdir(model_directory):
            os.makedirs(model_directory)
        self.model_directory = model_directory
        self.model_path = None
        self.optimizer = Adam(self.parameters())
        self.scheduler = ExponentialLR(self.optimizer, gamma=.99)
        self.scheduler.gamma = 0.9
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.epoch = 0
        if model_directory is not None:
            checkpoint = f'{base_name} (cp {checkpoint_index:04}).pth' if checkpoint_index is not None else None
            self.load(checkpoint=checkpoint)

        self.task = 'binary'
        self.threshold = .9
        self.threshold_tape = torch.arange(.05, 1.05, .05)
        self.metrics = {
            'Accuracy': Accuracy,
            'F1Score': F1Score,
            'Precision': Precision,
            'Recall': Recall,
            'Jaccard Index': JaccardIndex,
            'DSC': Dice
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward propagation through the whole model
        :param x: torch.Tensor, the images to be segmented
        :return: torch.Tensor, segmentation mask of the inputs
        """
        # forward propagate the inputs throughout the encoder and save the output of each block
        encoder_features = self.encoder(x)

        # forward propagate the encoder's reversed output through the decoder
        decoder_features = self.decoder(encoder_features[::-1][0], encoder_features[::-1][1:])

        # forward propagate the last output through the last layer to make shapes meet
        x = self.head_conv(decoder_features)

        # apply the activation function to classify the input
        x = self.head_activation(x)

        # if the user specifically wants the shapes to be preserved
        if self.retrain_dim:
            # interpolate the output shapes to match the desired size
            x = functional.interpolate(x, self.output_size)

        # return the masks with the desired shape
        return x

    def train_step(self,
                   images: torch.Tensor,
                   masks: torch.Tensor,
                   criterion: Callable,
                   optimizer: torch.optim.Optimizer,
                   learning_rate_scheduler: Any):
        """
        A single step of the training process.

        Parameters
        ----------
        images : torch.Tensor
            A tensor of images, shape (batch_size, C, H, W).
        masks : torch.Tensor
            A tensor of masks, shape (batch_size, C, H, W).
        criterion : Callable
            A loss function to evaluate the quality of the model's predictions.
        optimizer : torch.optim.Optimizer
            The optimizer used to update the model parameters.
        learning_rate_scheduler : Any
            A learning rate scheduler to adjust the learning rate during training.

        Returns
        -------
        loss : torch.Tensor
            The loss resulting from a single training step.
        """
        # Move the input tensors to the GPU for faster computation
        images, masks = (images.to(self.device), masks.to(self.device))

        # Use the model to make predictions on the input images
        predictions = self(images)

        # Compute the loss using the provided criterion
        loss = criterion(predictions, masks)

        # Zero the gradients in the optimizer
        optimizer.zero_grad()

        # Compute the gradients with respect to the model parameters
        loss.backward()

        # Update the model parameters using the computed gradients and the optimizer
        optimizer.step()

        # Update the learning rate using the learning rate scheduler
        learning_rate_scheduler.step()

        # return the resulted loss
        return loss, predictions

    def evaluate(self, data_loader, criterion):
        """
        Evaluate the performance of the model on the given data.

        Parameters:
        - data_loader (torch.utils.data.DataLoader): A DataLoader object containing the data to evaluate the model on.
        - criterion (callable): The loss function to use to evaluate the model.

        Returns:
        - float: The average loss of the model on the given data.
        """
        with torch.no_grad():
            self.eval()  # setting the model to evaluation mode
            total_loss = 0
            for images, masks in data_loader:
                images, masks = images.to(self.device), masks.to(self.device)  # moving the tensors to GPU, if available
                predictions = self(images)  # making predictions using the model
                total_loss += criterion(predictions, masks)  # accumulating the loss
        return total_loss / len(data_loader)  # returning the average loss

    def training_loop(self, **kwargs):
        """A function for training a PyTorch model.

        This function trains the model for a specified number of epochs and tracks
        the training and test losses during each epoch.

        Parameters:
        - **kwargs (dictionary): Keyword arguments containing the following:
            - 'train_data_loader': A PyTorch DataLoader object that provides training data
            - 'test_data_loader': A PyTorch DataLoader object that provides test data
            - 'criterion': A PyTorch loss function to be used in computing the training and test loss
            - 'num_epochs': An integer value representing the number of epochs to be run
            - 'checkpoints_interval': An integer value representing the frequency (in epochs) to save the model.

        Returns:
        - history (dictionary): A dictionary containing the following information:
            - 'training loss': A list of training loss values for each epoch
            - 'test loss': A list of test loss values for each epoch

        """
        # Set the model to training mode
        self.train()

        # Initialize a dictionary to store the history of training and test losses

        history = {'training loss': [],
                   'test loss': []
                   }
        history.update({key: [] for key in self.metrics.keys()})
        # Move the model to the GPU, if available
        self.to(self.device)

        # Initialize an Adam optimizer with a learning rate of 0.001
        optimizer = torch.optim.Adam(self.parameters(), lr=.0001)

        # Initialize a learning rate scheduler with an exponential decay rate of 0.99
        lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=.99)

        # Loop over the specified number of epochs
        for epoch in range(self.epoch, num_epochs, 1):

            # Initialize a variable to keep track of the total training loss during the epoch
            total_training_loss = 0

            # Use tqdm to display a progress bar
            with tqdm(total=len(kwargs['train_data_loader']),
                      desc=f'Epoch {epoch + 1}/{kwargs["num_epochs"]}') as progress_bar:

                # Loop over the training data in the data loader
                for index, (images, masks) in enumerate(kwargs['train_data_loader']):
                    # Compute the loss for the current batch of images and masks
                    loss, predictions = self.train_step(images=images,
                                                        masks=masks,
                                                        criterion=kwargs['criterion'],
                                                        optimizer=optimizer,
                                                        learning_rate_scheduler=lr_scheduler)

                    total_training_loss += loss
                    metrics = {key: metric(task=self.task,
                                           threshold=self.threshold)
                    (preds=predictions.detach().cpu(),
                     target=masks.detach().cpu().int())
                               for key, metric in self.metrics.items()}

                    [history[key].append(value) for key, value in metrics.items()]
                    metrics_str = ' | '.join([f'{key.capitalize()}: {value:.4f}'
                                              for key, value in metrics.items()])

                    # Update the progress bar
                    progress_bar.update(1)
                    progress_bar.set_postfix_str(f'Training Loss: {total_training_loss / (index + 1):.4f} | '
                                                 f'{metrics_str}')

                    if (index + 1) % kwargs['progress_plot_interval'] == 0:
                        with torch.no_grad():
                            jaccard_index = 0
                            for threshold in [x / 100 for x in range(5, 105, 5)]:
                                metrics = {key: metric(task=self.task,
                                                       threshold=threshold)
                                (preds=predictions.detach().cpu(),
                                 target=masks.detach().cpu().int())
                                           for key, metric in self.metrics.items()}
                                if metrics['Jaccard Index'] > jaccard_index:
                                    self.threshold = threshold

                            fig, ax = plt.subplots(3, 5, figsize=(15, 8))
                            for j in range(5):
                                ax[0][j].imshow(images[j].permute(1, 2, 0).cpu().numpy())
                                ax[1][j].imshow(np.where(predictions[j].cpu().numpy().squeeze() > self.threshold,
                                                         255,
                                                         0))
                                ax[2][j].imshow(masks[j].cpu().numpy().squeeze())
                            plt.show()

                # Compute the total test loss for the current epoch
                test_loss = self.evaluate(kwargs['test_data_loader'], criterion_)

                # Calculate the average training and test loss for the current epoch
                average_training_loss = total_training_loss / len(kwargs['train_data_loader'])

                # Append the average training and test loss to the history dict
                history['training loss'].append(average_training_loss)
                history['test loss'].append(test_loss)

                # Update the progress bar with the average training and test loss for the current epoch
                progress_bar.set_postfix({'Training Loss': f'{average_training_loss:.4f}',
                                          'Test Loss': f'{test_loss:.4f}'})

                self.epoch += 1
                # Save the model every 'checkpoints_interval' epochs
                if epoch % kwargs['checkpoints_interval'] == 0:
                    self.save()

        # Returns the final training and test loss history of the model
        return history

    def save(self):
        """
        This function saves the current state of the model, optimizer, loss, and scheduler.
        The model is saved with the following details:
            - The current epoch number
            - Model state dictionary
            - Optimizer state dictionary
            - Loss state dictionary
            - Scheduler state dictionary
        """
        # Create a dictionary to hold the state information
        state_dictionary = {
            'epoch': self.epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }

        # Construct the file path for saving the model
        model_path = os.path.join(self.model_directory, f'UNet (cp {self.epoch:04}).pth')

        # Save the state dictionary to the specified file path
        torch.save(state_dictionary, model_path)

    def load(self, checkpoint=None):
        """
        Load the saved model state.

        Args:
            checkpoint (str, optional): The checkpoint file to be loaded. If None, the latest checkpoint will be loaded.
        """
        # Create a list of checkpoint files in the model directory
        checkpoint_files = os.listdir(self.model_directory)

        # Check if there are any checkpoint files in the directory
        if checkpoint_files:
            # If checkpoint is not provided, retrieve the latest checkpoint
            if checkpoint is None:
                # Sort the checkpoint files by the checkpoint number
                checkpoint_files = sorted(checkpoint_files, key=lambda x: int(re.search(r'\(cp (\d+)\)', x).group(1)))
                checkpoint = checkpoint_files[-1]

            # Load the state dictionary from the checkpoint file
            checkpoint_path = os.path.join(self.model_directory, checkpoint)
            state_dictionary = torch.load(checkpoint_path)

            # Load the model parameters and other training-related objects
            self.epoch = state_dictionary['epoch']
            self.load_state_dict(state_dictionary['model_state_dict'])
            self.optimizer.load_state_dict(state_dictionary['optimizer_state_dict'])
            self.loss.load_state_dict(state_dictionary['loss'])
            self.scheduler.load_state_dict(state_dictionary['scheduler_state_dict'])
            try:
                self.threshold = state_dictionary['threshold']
            except KeyError:
                self.threshold = .5

    def infer(self, image):
        image = self(image).detach().permute([0, 2, 3, 1]).squeeze(dim=0).cpu()
        image = np.where(image > 0.5, 255, 0).astype(np.uint8)
        image = np.tile(image, [1, 1, 3])
        return image


if __name__ == '__main__':
    num_epochs = 100
    batch_size = 64
    checkpoints_interval = 5
    progress_plots_interval = 50
    criterion_ = JaccardLoss(mode='binary')
    train_data_loader = DataLoader(dataset=FireDataSegmentation(split='train'), batch_size=batch_size, shuffle=True)
    # valid_data_loader = DataLoader(dataset=FireDataSegmentation(split='valid'), batch_size=64, shuffle=True)
    test_data_loader = DataLoader(dataset=FireDataSegmentation(split='test'), batch_size=batch_size, shuffle=True)
    model = UNet(model_directory=os.path.join('Models', 'UNet Checkpoints'))
    model.training_loop(progress_plot_interval=progress_plots_interval,
                        train_data_loader=train_data_loader,
                        test_data_loader=test_data_loader,
                        num_epochs=num_epochs,
                        checkpoints_interval=checkpoints_interval,
                        criterion=criterion_)
