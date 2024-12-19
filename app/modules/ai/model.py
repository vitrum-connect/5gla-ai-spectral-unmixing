import torch.nn as nn
import torch


class CNNModel(nn.Module):
    @staticmethod
    def get_relu():
        return nn.ReLU()

    @staticmethod
    def get_conv(in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)  # Downsample by 2

    @classmethod
    def get_flattened_size(cls, test_input, complexity, num_layers):
        # Calculate the size after the convolutional layers
        with torch.no_grad():
            test_output = cls._get_conv_output(test_input, complexity, num_layers)
        return test_output.numel()

    @classmethod
    def _get_conv_output(cls, x, complexity, num_layers):
        relu = cls.get_relu()
        in_channels = x.shape[1]
        for layer_idx in range(num_layers):
            out_channels = 16 * complexity * (2 ** layer_idx)
            x = relu(cls.get_conv(in_channels, out_channels)(x))
            in_channels = out_channels
        return x

    @classmethod
    def from_test_input(cls, test_input, output_size, complexity=2, num_layers=4):
        flattened_size = cls.get_flattened_size(test_input, complexity, num_layers)
        input_shape = test_input.shape
        return cls(flattened_size, input_shape, output_size, complexity, num_layers)

    def __init__(self, flattened_size, input_shape, output_size, complexity=2, num_layers=4):
        super(CNNModel, self).__init__()
        self.complexity = complexity
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        self.relu = self.get_relu()

        # Dynamically create convolutional layers
        in_channels = input_shape[1]
        for layer_idx in range(num_layers):
            out_channels = 16 * complexity * (2 ** layer_idx)
            self.conv_layers.append(self.get_conv(in_channels, out_channels))
            in_channels = out_channels

        self.flattened_size = flattened_size
        self.output_size = output_size
        self.input_shape = input_shape

        self.fc1 = nn.Linear(self.flattened_size, 64 * complexity)  # Scale FC layer size with complexity
        self.fc2 = nn.Linear(64 * complexity, output_size)  # Output layer

    def forward(self, X):
        for conv in self.conv_layers:
            X = self.relu(conv(X))
        X = X.view(X.size(0), -1)  # Flatten the tensor for the fully connected layers
        X = self.relu(self.fc1(X))
        X = self.fc2(X)
        return X


def save_model(model, file_path):
    """
    Save the model state_dict along with flattened_size in the metadata.

    Args:
        model (nn.Module): The trained model instance.
        file_path (str): Path to save the model.
    """
    # Collect model state and metadata
    metadata = {
        'state_dict': model.state_dict(),
        'flattened_size': model.flattened_size,
        'output_size': model.output_size,
        'input_shape': model.input_shape,
        'complexity': model.complexity,
        'num_layers': model.num_layers,
    }
    # Save to file
    torch.save(metadata, file_path)
    print(f"Model and metadata saved to {file_path}")
