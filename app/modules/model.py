import torch.nn as nn
import torch

class CNNModel(nn.Module):

    @staticmethod
    def get_relu():
        return nn.ReLU()

    @staticmethod
    def get_conv_2():
        return nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # Downsample by 2 again

    @staticmethod
    def get_conv_1(nr_channels):
        return nn.Conv2d(nr_channels, 16, kernel_size=3, stride=2, padding=1)  # Downsample by 2

    @classmethod
    def get_flattened_size(cls, test_input):
        # Calculate the size after the convolutional layers
        with torch.no_grad():
            test_output = cls._get_conv_output(test_input)
        return test_output.numel()

    @classmethod
    def _get_conv_output(cls, x):
        relu = cls.get_relu()
        x = relu(cls.get_conv_1(x.shape[1])(x))
        x = relu(cls.get_conv_2()(x))
        return x

    @classmethod
    def from_test_input(cls, test_input, output_size):
        flattened_size = cls.get_flattened_size(test_input)
        input_shape = test_input.shape
        return cls(flattened_size, input_shape, output_size)

    def __init__(self, flattened_size, input_shape, output_size):
        super(CNNModel, self).__init__()
        self.conv1 = self.get_conv_1(input_shape[1])
        self.conv2 = self.get_conv_2()
        self.relu = self.get_relu()
        self.flattened_size = flattened_size
        self.output_size = output_size
        self.input_shape = input_shape

        self.fc1 = nn.Linear(self.flattened_size, 64)  # Adjust based on flattened size
        self.fc2 = nn.Linear(64, output_size)  # Output layer

    def forward(self, X):
        X = self.relu(self.conv1(X))
        X = self.relu(self.conv2(X))
        X = X.view(X.size(0), -1)  # Flatten the tensor for the fully connected layers
        X = self.relu(self.fc1(X))
        X = self.fc2(X)
        return X
