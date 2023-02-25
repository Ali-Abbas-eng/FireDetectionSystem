import os.path
import numpy as np
from keras import layers
import tensorflow as tf
from typing import Callable
import keras.backend as keras_backend


class DownscaleBlock(layers.Layer):
    """
    DownscaleBlock is a layer which represents a single down-sampling block of a convolutional neural network.
    It applies two
    convolutional filters, ReLU activation,
    and batch normalization to the input tensor followed by a max pooling operation.

    Args:
    filters (int): number of filters in each convolutional layer
    kernel_size (Tuple[int, int]): size of the convolutional kernel
    padding (str): type of padding used in convolution
    strides (int): stride of convolution operation

    Returns:
    output (Tuple[Tensor, Tensor]): a tuple of the output tensor and max pooled tensor
    """
    def __init__(
            self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        # First convolutional filter
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        # Second convolutional filter
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        # ReLU activation function
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)
        # Batch normalization
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.bn2b = tf.keras.layers.BatchNormalization()
        # Max pooling operation
        self.pool = layers.MaxPool2D((2, 2), (2, 2))

    def call(self, input_tensor, *args, **kwargs):
        """
        Apply the convolutional filters, ReLU activation, and batch normalization to the input tensor, and then perform
        max pooling.

        Args:
        input_tensor (Tensor): input tensor to be processed

        Returns:
        output (Tuple[Tensor, Tensor]): a tuple of the output tensor and max pooled tensor
        """
        d = self.convA(input_tensor)
        x = self.bn2a(d)
        x = self.reluA(x)

        x = self.convB(x)
        x = self.bn2b(x)
        x = self.reluB(x)

        x += d
        p = self.pool(x)
        return x, p


class UpscaleBlock(layers.Layer):
    """
    A class representing an upscale block for use in an image super-resolution model.

    The block consists of an upsampling layer, a concatenation layer, two convolutional layers,
    two leaky ReLU activation layers, and two batch normalization layers.

    Attributes:
        us: An upsample layer for increasing the dimensions of the input tensor.
        convA: A convolutional layer that takes in a tensor and applies a 2D convolution to it.
        convB: A second convolutional layer that takes in a tensor and applies a 2D convolution to it.
        reluA: A leaky ReLU activation layer for introducing non-linearity in the model.
        reluB: A second leaky ReLU activation layer.
        batch_normalisation_2a: A batch normalization layer that normalizes the input tensor's values.
        batch_normalisation_2b: A second batch normalization layer.
        concatenation: A concatenation layer that combines two tensors along a specified axis.

    Methods:
        call(x, *args, **kwargs):
            Calls the upscale block on an input tensor x.

    """
    def __init__(
            self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        # Initialize layers with given parameters
        self.us = layers.UpSampling2D((2, 2))
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)
        self.batch_normalisation_2a = tf.keras.layers.BatchNormalization()
        self.batch_normalisation_2b = tf.keras.layers.BatchNormalization()
        self.concatenation = layers.Concatenate()

    def call(self, x, *args, **kwargs):
        """
        Calls the upscale block on an input tensor x.

        Args:
            x: The input tensor to the block.
            *args: Additional positional arguments to be passed to the call method.
            **kwargs: Additional keyword arguments to be passed to the call method.

        Returns:
            The output tensor of the block.

        """
        x, skip = x  # Split x into two tensors
        x = self.us(x)  # Upsample x using the upsample layer
        concat = self.concatenation([x, skip])  # Concatenate x with the skip connection tensor
        x = self.convA(concat)  # Apply the first convolutional layer
        x = self.batch_normalisation_2a(x)  # Normalize the output of the first convolutional layer
        x = self.reluA(x)  # Apply the first leaky ReLU activation
        x = self.convB(x)  # Apply the second convolutional layer
        x = self.batch_normalisation_2b(x)  # Normalize the output of the second convolutional layer
        x = self.reluB(x)  # Apply the second leaky ReLU activation
        return x  # Return the output tensor


class BottleNeckBlock(layers.Layer):
    def __init__(
            self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        """
        Initializes the Bottleneck Block Layer.

        Args:
            filters: (int) number of convolutional filters to be used in the convolution layers
            kernel_size: (tuple of 2 integers) the height and width of the 2D convolution window
            padding: (str) one of 'valid' or 'same' (case-insensitive)
            strides: (int or tuple of 2 integers) specifying the strides of the convolution along the height and width
                Default value is 1
        """
        super().__init__(**kwargs)
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)

    def call(self, x, *args, **kwargs):
        """
        Defines the computation performed by the bottleneck block layer.

        Args:
            x: input tensor
            args: additional positional arguments to be passed to the method
            kwargs: additional keyword arguments to be passed to the method

        Returns:
            x: tensor obtained by performing convolution operation with two convolution layers and applying LeakyReLU
        """
        x = self.convA(x)
        x = self.reluA(x)
        x = self.convB(x)
        x = self.reluB(x)
        return x


def depth_loss_function(target, pred, **kwargs):
    """Calculate the composite depth estimation loss between the target and predicted tensors.

    Args:
        target: A tensor representing the ground truth depth map.
        pred: A tensor representing the predicted depth map.
        **kwargs: Optional keyword arguments that specify the weighting of different loss components.

    Returns:
        A scalar tensor representing the total depth estimation loss between the target and predicted tensors.
    """
    # Extract the relative weighting of different loss components
    ssim_loss_weight = kwargs.get('ssim', .9)
    l1_loss_weight = kwargs.get('l1', .1)
    edge_loss_weight = kwargs.get('edge', .8)

    # Compute the image gradients for both the target and predicted tensors
    dy_true, dx_true = tf.image.image_gradients(target)
    dy_pred, dx_pred = tf.image.image_gradients(pred)

    # Compute the weights for the edge-based depth smoothness loss
    weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
    weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

    # Compute the smoothed gradients for the predicted depth map
    smoothness_x = dx_pred * weights_x
    smoothness_y = dy_pred * weights_y

    # Compute the depth smoothness loss for the predicted depth map
    depth_smoothness_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))

    # Compute the structural similarity (SSIM) index between the target and predicted depth maps
    ssim_loss = tf.reduce_mean(1 - tf.image.ssim(target, pred, max_val=256, filter_size=7, k1=0.01 ** 2, k2=0.03 ** 2))

    # Compute the point-wise L1 loss between the target and predicted depth maps
    l1_loss = tf.reduce_mean(tf.abs(target - pred))

    # Combine the individual loss components into a single composite loss value
    loss = ((ssim_loss_weight * ssim_loss) + (l1_loss_weight * l1_loss) + (edge_loss_weight * depth_smoothness_loss))

    # Return the total loss value
    return loss


def dice_bce_loss(targets, inputs, smooth=1e-6):
    """This function calculates the dice and binary cross-entropy loss for binary classification.

    Args:
        targets: tensor, the true labels for the inputs. The shape should be (batch_size, height, width, depth)
        inputs: tensor, the predicted labels for the inputs. The shape should be (batch_size, height, width, depth)
        smooth: a very small float number to avoid dividing by zero error in the loss calculation.

    Returns:
        dice_bce: tensor, the total loss, which is the sum of the binary cross-entropy loss and the dice loss.

    """
    inputs = keras_backend.flatten(inputs)
    targets = keras_backend.flatten(targets)

    # binary cross-entropy loss
    bce = tf.keras.losses.binary_crossentropy(targets, inputs)

    # dice loss
    intersection = keras_backend.sum(targets * inputs)
    dice_loss = 1 - (2 * intersection + smooth) / (keras_backend.sum(targets) + keras_backend.sum(inputs) + smooth)

    # dice and binary cross-entropy loss
    dice_bce = bce + dice_loss

    return dice_bce


class UNet(tf.keras.Model):
    def __init__(self,
                 model_directory: str = os.path.join('Models', 'UNet Depth Estimation'),
                 file_name: str = 'checkpoint.h5',
                 loss_function: Callable = depth_loss_function,
                 top_layer_activation: str = 'relu',
                 squeeze_outputs: bool = False,
                 **kwargs):
        """
        The constructor of the UNet class.

        Args:
        - model_directory (str): The directory where the model will be saved.
        - file_name (str): The file name to save the model.
        - loss_function (Callable): The function used to calculate the loss.
        - top_layer_activation (str): The activation function to use for the top layer.
        - **kwargs: Additional arguments.

        Returns:
        - None.
        """
        super().__init__()
        self.squeeze_outputs = squeeze_outputs
        self.kwargs = kwargs
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.loss_function = loss_function
        # Filters to use for the downscale, bottleneck, and upscale blocks.
        filters = [16, 32, 64, 128, 256]
        # Create a DownscaleBlock for each filter in filters except the last one.
        self.downscale_blocks = [DownscaleBlock(config) for config in filters[:-1]]
        # Create a BottleNeckBlock for the last filter.
        self.bottle_neck_block = BottleNeckBlock(filters[-1])
        # Create an UpscaleBlock for each filter in filters starting from the second-last filter.
        self.upscale_blocks = [UpscaleBlock(config) for config in filters[-2::-1]]
        # The final convolution layer.
        self.conv_layer = layers.Conv2D(1, (1, 1), padding="same")
        # The activation layer for the top layer.
        self.activation = layers.Activation(top_layer_activation)
        # Create the model directory if it does not exist.
        model_directory = model_directory
        if not os.path.isdir(model_directory):
            os.makedirs(model_directory)
        # The path to save the model.
        self.file_path = os.path.join(model_directory, file_name)
        # The ModelCheckpoint callback to save the model.
        self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.file_path,
                                                                      monitor='val_loss',
                                                                      save_weights_only=True,
                                                                      save_best_only=True,
                                                                      mode='min',
                                                                      save_freq='epoch')
        # The TensorBoard callback to log the training progress.
        self.tensor_board_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_directory, 'logs'),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
        )

    @property
    def metrics(self):
        """
        A property that returns a list of metrics, in this case just the loss metric.
        """
        return [self.loss_metric]

    def calculate_loss(self, target, pred, **kwargs):
        """
        A method to calculate the loss for the given target and predicted values.

        :param target: The target values to calculate the loss.
        :param pred: The predicted values to calculate the loss.
        :param kwargs: Additional keyword arguments.
        :return: loss.
        """
        return self.loss_function(target, pred, **kwargs)

    def train_step(self, batch_data):
        """
        A method that is called for each training batch. It computes the loss, gradients, and applies the gradients
        to update the weights of the network.

        :param batch_data: The input batch data.
        :return: A dictionary containing the loss value.
        """
        inputs, target = batch_data

        # Gradient tape to calculate the gradients of the loss with respect to the trainable variables
        with tf.GradientTape() as tape:
            pred = self(inputs, training=True)

            if self.squeeze_outputs:
                pred = tf.squeeze(pred)

            loss = self.calculate_loss(target, pred, **self.kwargs)
        gradients = tape.gradient(loss, self.trainable_variables)

        # Apply the gradients to update the trainable variables
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update the loss metric with the calculated loss value
        self.loss_metric.update_state(loss)

        # Return a dictionary containing the loss value
        return {"loss": self.loss_metric.result()}

    def test_step(self, batch_data):
        """
        Performs a single evaluation step on a batch of data.

        Args:
            batch_data: A tuple of (inputs, targets).

        Returns:
            A dictionary containing the loss value.
        """
        inputs, target = batch_data

        # Forward pass through the network.
        pred = self(inputs, training=False)

        if self.squeeze_outputs:
            # remove redundant axis
            pred = tf.squeeze(pred)

        # Calculate the loss value.
        loss = self.calculate_loss(target, pred)

        # Update the loss metric.
        self.loss_metric.update_state(loss)

        return {
            "loss": self.loss_metric.result(),
        }

    def call(self, x, *args, **kwargs):
        """
        Implements the forward pass of the UNet model.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        # Pass the input through the down-sampling blocks.
        c1, p1 = self.downscale_blocks[0](x)
        c2, p2 = self.downscale_blocks[1](p1)
        c3, p3 = self.downscale_blocks[2](p2)
        c4, p4 = self.downscale_blocks[3](p3)

        # Pass the output of the last down-sampling block through the bottleneck block.
        bn = self.bottle_neck_block(p4)

        # Pass the bottleneck block output through the upsampling blocks in reverse order.
        u1 = self.upscale_blocks[0]([bn, c4])
        u2 = self.upscale_blocks[1]([u1, c3])
        u3 = self.upscale_blocks[2]([u2, c2])
        u4 = self.upscale_blocks[3]([u3, c1])

        # Pass the output of the last upsampling block through the final convolution layer.
        return self.conv_layer(u4)


def get_depth_estimation_model(model_directory: str = os.path.join('Models', 'UNet Depth Estimation'),
                               weights_model: str = os.path.join('Models', 'UNet Depth Estimation', 'checkpoint.h5'),
                               file_name: str = 'checkpoint.h5',
                               loss_function: Callable = depth_loss_function,
                               top_layer_activation: str = 'relu',
                               assert_file_exists: bool = False,
                               **kwargs) -> UNet:
    """
    Returns a pre-trained or a newly initialized UNet model for depth estimation.

    Args:
        model_directory (str, optional): The directory to load/save the model.
                Defaults to 'Models/UNet Depth Estimation'.
        weights_model (str, optional): The path to the weights file to load.
                Defaults to 'Models/UNet Depth Estimation/checkpoint.h5'.
        file_name (str, optional): The name of the weights file to save.
                Defaults to 'checkpoint.h5'.
        loss_function (Callable, optional): The loss function to use in training the model.
                Defaults to calculate_loss_depth_estimation.
        top_layer_activation (str, optional): The activation function to use in the final layer of the model.
                Defaults to 'relu'.
        assert_file_exists (bool, optional): whether to throw an error in case there is no saved checkpoint.
                Defaults to False
        **kwargs: Additional keyword arguments to pass to the UNet constructor.

    Returns:
        model (UNet): The pre-trained or initialized UNet model.
    """
    model = UNet(model_directory=model_directory,
                 weights_model=weights_model,
                 file_name=file_name,
                 loss_function=loss_function,
                 top_layer_activation=top_layer_activation,
                 **kwargs)

    # Test the model with a random input
    model(np.random.normal(size=(2, 256, 256, 3)))

    # Load the saved weights if the file exists
    if os.path.isfile(model.file_path):
        model.load_weights(model.file_path)
    elif assert_file_exists:
        raise FileNotFoundError('No Saved Checkpoint From Which We Can Load Trained Weights')

    return model


def get_fire_segmentation_model(model_directory: str = os.path.join('Models', 'UNet Fire Segmentation'),
                                file_name: str = 'checkpoint.h5',
                                loss_function: Callable = dice_bce_loss,
                                top_layer_activation: str = 'sigmoid',
                                assert_file_exists: bool = False,
                                **kwargs) -> tf.keras.Model:
    """
    Returns a pre-trained UNet model for fire segmentation.

    Args:
        model_directory (str): The directory where the model is stored. Default is 'Models/UNet Fire Segmentation'.
        file_name (str): The name of the checkpoint file. Default is 'checkpoint.h5'.
        loss_function (Callable): The loss function used to train the model. Default is dice_bce_loss.
        top_layer_activation (str): The activation function used in the final layer. Default is 'sigmoid'.
        assert_file_exists (bool, optional): whether to throw an error in case there is no saved checkpoint.
                Defaults to False
        **kwargs: Additional keyword arguments to be passed to the UNet constructor.

    Returns:
        A pre-trained UNet model for fire segmentation.
    """

    # Create a UNet model instance
    model = UNet(model_directory=model_directory,
                 file_name=file_name,
                 loss_function=loss_function,
                 top_layer_activation=top_layer_activation,
                 squeeze_outputs=True,
                 **kwargs)

    # Check the model using a random input
    model(np.random.normal(size=(2, 256, 256, 3)))

    # Load the pre-trained weights if the checkpoint file exists
    if os.path.isfile(model.file_path):
        model.load_weights(model.file_path)
    elif assert_file_exists:
        raise FileNotFoundError('No Saved Checkpoint From Which We Can Load Trained Weights')

    return model
