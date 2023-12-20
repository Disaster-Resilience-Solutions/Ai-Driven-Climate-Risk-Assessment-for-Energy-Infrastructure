# Wind Speed Component Forecasting Convolutional Neural Network (CNN) Model

# This file defines a CNN model for wind forecasting, consisting of custom layers for handling periodic padding
# and convolution operations. The model architecture is built using TensorFlow and Keras.

# Custom Layers:
#     - PeriodicPadding2D: Periodic padding layer for 2D inputs, addressing boundary conditions in convolutional layers.
#     - PeriodicConv2D: Periodic convolutional layer for 2D inputs, applying periodic convolution to handle boundary conditions.

# Model Building Function:
#     - build_cnn: Function to build the CNN model based on specified filters, kernels, input shape, and optional dropout.

# Usage:
#     1. Import the necessary modules: TensorFlow, Keras, numpy, and xarray.
#     2. Utilize the provided custom layers to handle periodic padding and convolution.
#     3. Use the build_cnn function to create the wind forecasting CNN model with the desired architecture.

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, Dropout
from tensorflow.keras.models import Model

class PeriodicPadding2D(tf.keras.layers.Layer):
    """
    Periodic Padding Layer for 2D inputs.

    This layer pads the input tensor periodically to handle boundary conditions in convolutional layers.

    Args:
        pad_width (int): The width of padding to be applied to the input tensor.

    Attributes:
        pad_width (int): The width of padding to be applied to the input tensor.

    Usage:
        periodic_padding = PeriodicPadding2D(pad_width=1)
        x_padded = periodic_padding(input_tensor)
    """

    def __init__(self, pad_width, **kwargs):
        super().__init__(**kwargs)
        self.pad_width = pad_width

    def call(self, inputs, **kwargs):
        """
        Apply periodic padding to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Tensor with periodic padding applied.
        """
        if self.pad_width == 0:
            return inputs
        inputs_padded = tf.concat(
            [inputs[:, :, -self.pad_width:, :], inputs, inputs[:, :, :self.pad_width, :]], axis=2)
        inputs_padded = tf.pad(inputs_padded, [[0, 0], [self.pad_width, self.pad_width], [0, 0], [0, 0]])
        return inputs_padded

    def get_config(self):
        """
        Get the configuration dictionary for the layer.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config()
        config.update({'pad_width': self.pad_width})
        return config

class PeriodicConv2D(tf.keras.layers.Layer):
    """
    Periodic Convolutional Layer for 2D inputs.

    This layer applies periodic convolution to the input tensor.

    Args:
        filters (int): The number of filters in the convolution.
        kernel_size (int or tuple): The size of the convolution kernel.
        conv_kwargs (dict): Additional keyword arguments for the Conv2D layer.

    Attributes:
        filters (int): The number of filters in the convolution.
        kernel_size (int or tuple): The size of the convolution kernel.
        conv_kwargs (dict): Additional keyword arguments for the Conv2D layer.
        padding (PeriodicPadding2D): PeriodicPadding2D layer for handling boundary conditions.
        conv (Conv2D): Conv2D layer for the convolution operation.

    Usage:
        periodic_conv = PeriodicConv2D(filters=64, kernel_size=3)
        output_tensor = periodic_conv(input_tensor)
    """

    def __init__(self, filters,
                 kernel_size,
                 conv_kwargs={},
                 **kwargs, ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv_kwargs = conv_kwargs
        if type(kernel_size) is not int:
            assert kernel_size[0] == kernel_size[1], 'PeriodicConv2D only works for square kernels'
            kernel_size = kernel_size[0]
        pad_width = (kernel_size - 1) // 2
        self.padding = PeriodicPadding2D(pad_width)
        self.conv = Conv2D(filters, kernel_size, padding='valid', **conv_kwargs)

    def call(self, inputs):
        """
        Apply periodic convolution to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Tensor after applying periodic convolution.
        """
        return self.conv(self.padding(inputs))

    def get_config(self):
        """
        Get the configuration dictionary for the layer.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config()
        config.update({'filters': self.filters, 'kernel_size': self.kernel_size, 'conv_kwargs': self.conv_kwargs})
        return config

def build_cnn(filters, kernels, input_shape, dr=0):
    """
    Build a Convolutional Neural Network (CNN) model.

    Args:
        filters (list): List of integers, specifying the number of filters for each convolutional layer.
        kernels (list): List of integers or tuples, specifying the kernel size for each convolutional layer.
        input_shape (tuple): Shape of the input tensor (excluding batch size).
        dr (float): Dropout rate, if greater than 0.

    Returns:
        tf.keras.models.Model: CNN model.
    """
    x = input = Input(shape=input_shape)

    for f, k in zip(filters[:-1], kernels[:-1]):
        x = PeriodicConv2D(f, k)(x)
        x = LeakyReLU()(x)
        if dr > 0: x = Dropout(dr)(x)

    output = PeriodicConv2D(filters[-1], kernels[-1])(x)

    return Model(input, output)
