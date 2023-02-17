import os.path

from keras import layers
import tensorflow as tf
from typing import Callable
import keras.backend as keras_backend


class DownscaleBlock(layers.Layer):
    def __init__(
            self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.pool = layers.MaxPool2D((2, 2), (2, 2))

    def call(self, input_tensor, *args, **kwargs):
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
    def __init__(
            self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.us = layers.UpSampling2D((2, 2))
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.conc = layers.Concatenate()

    def call(self, x, *args, **kwargs):
        x, skip = x
        x = self.us(x)
        concat = self.conc([x, skip])
        x = self.convA(concat)
        x = self.bn2a(x)
        x = self.reluA(x)

        x = self.convB(x)
        x = self.bn2b(x)
        x = self.reluB(x)

        return x


class BottleNeckBlock(layers.Layer):
    def __init__(
            self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)

    def call(self, x, *args, **kwargs):
        x = self.convA(x)
        x = self.reluA(x)
        x = self.convB(x)
        x = self.reluB(x)
        return x


def calculate_loss_depth_estimation(target, pred, **kwargs):
    ssim_loss_weight = kwargs.get('ssim', .9)
    l1_loss_weight = kwargs.get('l1', .1)
    edge_loss_weight = kwargs.get('edge', .8)
    # Edges
    dy_true, dx_true = tf.image.image_gradients(target)
    dy_pred, dx_pred = tf.image.image_gradients(pred)
    weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
    weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

    # Depth smoothness
    smoothness_x = dx_pred * weights_x
    smoothness_y = dy_pred * weights_y

    depth_smoothness_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(
        abs(smoothness_y)
    )

    # Structural similarity (SSIM) index
    ssim_loss = tf.reduce_mean(
        1
        - tf.image.ssim(
            target, pred, max_val=256, filter_size=7, k1=0.01 ** 2, k2=0.03 ** 2
        )
    )
    # Point-wise depth
    l1_loss = tf.reduce_mean(tf.abs(target - pred))

    loss = (
            (ssim_loss_weight * ssim_loss)
            + (l1_loss_weight * l1_loss)
            + (edge_loss_weight * depth_smoothness_loss)
    )
    return loss


class UNet(tf.keras.Model):
    def __init__(self,
                 model_directory: str = os.path.join('Models', 'UNet Depth Estimation'),
                 file_name: str = 'checkpoint.h5',
                 loss_function: Callable = calculate_loss_depth_estimation,
                 top_layer_activation: str = 'relu',
                 **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.calculate_loss = loss_function
        filters = [16, 32, 64, 128, 256]
        self.downscale_blocks = [DownscaleBlock(config) for config in filters[:-1]]
        self.bottle_neck_block = BottleNeckBlock(filters[-1])
        self.upscale_blocks = [UpscaleBlock(config) for config in filters[-2::-1]]
        self.conv_layer = layers.Conv2D(1, (1, 1), padding="same", activation="relu")
        model_directory = model_directory
        if not os.path.isdir(model_directory):
            os.makedirs(model_directory)
        self.file_path = os.path.join(model_directory, file_name)
        if self.conv_layer.activation != top_layer_activation:
            self.conv_layer = layers.Conv2D(1, (1, 1), padding="same", activation="relu")

        self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.file_path,
                                                                      monitor='val_loss',
                                                                      save_weights_only=True,
                                                                      save_best_only=True,
                                                                      mode='min',
                                                                      save_freq='epoch')
        self.tensor_board_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_directory, 'logs'),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
        )

    @property
    def metrics(self):
        return [self.loss_metric]

    def calculate_loss(self, target, pred):
        return calculate_loss_depth_estimation(target, pred, **self.kwargs)

    def train_step(self, batch_data):
        inputs, target = batch_data
        with tf.GradientTape() as tape:
            pred = self(inputs, training=True)
            loss = self.calculate_loss(target, pred, **self.kwargs)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_metric.update_state(loss)
        return {
            "loss": self.loss_metric.result(),
        }

    def test_step(self, batch_data):
        inputs, target = batch_data

        pred = self(inputs, training=False)
        loss = self.calculate_loss(target, pred)

        self.loss_metric.update_state(loss)
        return {
            "loss": self.loss_metric.result(),
        }

    def call(self, x, *args, **kwargs):
        c1, p1 = self.downscale_blocks[0](x)
        c2, p2 = self.downscale_blocks[1](p1)
        c3, p3 = self.downscale_blocks[2](p2)
        c4, p4 = self.downscale_blocks[3](p3)

        bn = self.bottle_neck_block(p4)

        u1 = self.upscale_blocks[0]([bn, c4])
        u2 = self.upscale_blocks[1]([u1, c3])
        u3 = self.upscale_blocks[2]([u2, c2])
        u4 = self.upscale_blocks[3]([u3, c1])

        return self.conv_layer(u4)


def dice_coefficient(y_true, y_pred, **kwargs):
    smooth = kwargs.get('smooth', 100)
    y_true_f = keras_backend.flatten(y_true)
    y_pred_f = keras_backend.flatten(y_pred)
    intersection = keras_backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (keras_backend.sum(y_true_f) + keras_backend.sum(y_pred_f) + smooth)
    return dice


def get_depth_estimation_model(model_directory: str = os.path.join('Models', 'UNet Depth Estimation'),
                               weights_model: str = os.path.join('Models', 'UNet Depth Estimation', 'checkpoint.h5'),
                               file_name: str = 'checkpoint.h5',
                               loss_function: Callable = calculate_loss_depth_estimation,
                               top_layer_activation: str = 'relu',
                               **kwargs):
    return UNet(model_directory=model_directory,
                weights_model=weights_model,
                file_name=file_name,
                loss_function=loss_function,
                top_layer_activation=top_layer_activation,
                **kwargs)


def get_fire_segmentation_model(model_directory: str = os.path.join('Models', 'UNet Fire Segmentation'),
                                file_name: str = 'checkpoint.h5',
                                loss_function: Callable = dice_coefficient,
                                top_layer_activation: str = 'sigmoid',
                                **kwargs):
    return UNet(model_directory=model_directory,
                file_name=file_name,
                loss_function=loss_function,
                top_layer_activation=top_layer_activation,
                **kwargs)
