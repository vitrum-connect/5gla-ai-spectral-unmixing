from tifffile import TiffFile

from app.modules.dataset import MultispectralDataset
from app.modules.model import CNNModel
from app.paths_handler import PathsManager
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tifffile import imread
import numpy as np
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta


def read_soil_moisture():
    df = pd.read_csv(r'../../data/Bodenfeuchte/Ostfalia011.csv')
    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%Y/%m/%d %H:%M:%S')
    # Remove rows with negative values in any "A.." columns
    a_columns = [col for col in df.columns if col.startswith('A')]
    df = df[(df[a_columns] >= 0).all(axis=1)]
    return df

def create_plot(df):
    colors = itertools.cycle(plt.cm.tab10.colors)
    plt.figure(figsize=(10, 6))
    for i in range(9):
        field_name = f'A{i+1}({5 + 10*i})'
        plt.plot(df['Date Time'], df[field_name], label=field_name, color=next(colors))

    # Add labels, legend, and title
    plt.xlabel('Date Time')
    plt.ylabel('Values')
    plt.title('Time Series Plot of A1 and A2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()


def get_dates_to_objects(psis, bucket_name):
    dates_to_objects = {}
    for obj in psis._list_files_in_bucket(bucket_name):
        if "KI-Use-Case" not in obj.object_name:
            continue
        try:
            capture_date = psis.get_time_of_capture(bucket_name, obj.object_name)
            capture_date = datetime.strptime(capture_date, "%Y:%m:%d %H:%M:%S")
            dates_to_objects[capture_date] = obj
        except Exception as e:
            print(f"Error processing {obj.object_name}: {e}")
            continue
    return dates_to_objects

def download_closest_image(psis, bucket_name, given_date, dates_to_objects):
    # Ensure `given_date` is a datetime object
    # given_date = datetime.strptime(given_date, "%Y/%m/%d %H:%M:%S")  # Adjust format as needed
    dates = list(dates_to_objects.keys())
    date_closest = min(dates, key=lambda d: abs(d - given_date))
    closest_object = dates_to_objects[date_closest]

    # Download the closest image
    if closest_object:
        pm = PathsManager(closest_object)
        for fname in pm.file_paths_stationary:
            try:
                response = psis.client.get_object(bucket_name, fname)

                path = "/".join(fname.split("/")[0:-1])
                os.makedirs(path, exist_ok=True)
                with open(fname, "wb") as f:
                    for chunk in response.stream(32 * 1024):  # Stream in chunks of 32 KB
                        f.write(chunk)

                print(f"Closest image downloaded to: {fname}")
            except Exception as e:
                print(f"Error downloading the image {fname}: {e}")
    else:
        print("No images found.")

# Define a custom dataset class
def crop_images(images):
    min_height = min(img.shape[0] for img in images)
    min_width = min(img.shape[1] for img in images)

    # 2. Crop all images to the smallest dimensions
    cropped_images = []
    for img in images:
        cropped = img[:min_height, :min_width, ...]  # Crop height and width
        cropped_images.append(cropped)
    return cropped_images


# Define a function to extract capture times and match to data
def load_images_and_labels(image_folder_1, image_folder_2, df, assumed_capillarity=0.5,
                           measure_depths=(5, 15, 25, 35, 45, 55, 65, 75, 85)):
    """
    Load pairs of images and corresponding labels from two image folders.

    Args:
        image_folder_1: Path to the first image folder.
        image_folder_2: Path to the second image folder (must have the same structure).
        df: DataFrame containing timestamps and measurement values.
        assumed_capillarity: Speed water rises in plant (cm per minute).
        measure_depths: Depths at which moisture measurements are taken.

    Returns:
        images: A list of tuples containing paired images (image_1, image_2).
        labels: A list of labels corresponding to moisture measurements.
    """
    images = []
    labels = []

    for root, _, files in os.walk(image_folder_1):  # Recursively go through all folders in folder 1
        for file_name in files:
            if file_name.endswith('.tif'):
                file_path_1 = os.path.join(root, file_name)

                # Construct corresponding path in image_folder_2
                relative_path = os.path.relpath(file_path_1, image_folder_1)
                splitted = relative_path.split(".")
                splitted[-2] += "_savi"
                relative_path = ".".join(splitted)
                file_path_2 = os.path.join(image_folder_2, relative_path)

                # Ensure the corresponding image exists in image_folder_2
                if not os.path.exists(file_path_2):
                    print(f"Warning: {file_path_2} not found, skipping.")
                    continue

                # Load both images
                image_1 = imread(file_path_1)
                image_2 = imread(file_path_2)

                # Extract capture_time metadata from the first image
                with TiffFile(file_path_1) as tif:
                    tags = tif.pages[0].tags
                    capture_time = tags.get('DateTime', None)
                    capture_time = capture_time.value if capture_time else None

                # Ensure capture_time is a datetime object for comparison
                capture_time = pd.to_datetime(capture_time, format='%Y:%m:%d %H:%M:%S')

                # Calculate moisture values
                moistures = []
                for i, measure_depth in enumerate(measure_depths):
                    delta_time_minutes = measure_depth / assumed_capillarity
                    delta_time = timedelta(minutes=delta_time_minutes)
                    row_name = f'Time Difference({measure_depth})'
                    df[row_name] = (df['Date Time'] - delta_time - capture_time).abs()
                    # Find the row with the minimum time difference
                    idx_min = df[row_name].idxmin()
                    matched_row = df.loc[idx_min]
                    moisture = matched_row[f"A{i + 1}({measure_depth})"]
                    moistures.append(moisture)

                # Append the image pair and moisture values
                images.append(np.dstack((image_1, image_2)))
                labels.append(np.array(moistures))

    return images, labels


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: The CNN model.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer for model parameters.
        device: Computation device.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def train_torch(image_folder, image_folder2, df, batch_size=16, num_epochs=10,
                learning_rate=0.001, test_size=0.2, random_state=42,
                model_save_path='multispectral_model.pth', assumed_capillarity=0.5, verbose=False):
    """
    Train a CNN model on multispectral image data.

    Args:
        image_folder (str): Path to the first image folder.
        image_folder2 (str): Path to the second image folder.
        df (pd.DataFrame): DataFrame containing moisture data.
        batch_size (int): Batch size for training and validation.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        test_size (float): Fraction of data to use for testing.
        random_state (int): Random seed for train-test split.
        model_save_path (str): Path to save the trained model.
    """
    device = get_device()
    images, labels = prepare_data(image_folder, image_folder2, df, assumed_capillarity)
    # labels /= np.max(labels)
    # print(labels)
    train_loader, test_loader = create_dataloaders(images, labels, batch_size, test_size, random_state)

    model = initialize_model(images, labels, device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    val_mse_all = []
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_mse = validate_model(model, test_loader, device)
        val_mse_all.append(val_mse)
        if verbose:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Validation MSE: {val_mse:.4f}")

    save_model(model, model_save_path)
    return min(val_mse_all)


def get_device():
    """Get the computation device: GPU or CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def prepare_data(image_folder, image_folder2, df, assumed_capillarity=0.5):
    """
    Load images, crop them, and prepare labels.

    Args:
        image_folder (str): Path to the first image folder.
        image_folder2 (str): Path to the second image folder.
        df (pd.DataFrame): DataFrame containing moisture data.

    Returns:
        images (ndarray): Cropped and prepared image data.
        labels (ndarray): Corresponding labels.
    """
    # Load and process images and labels
    images, labels = load_images_and_labels(image_folder, image_folder2, df, assumed_capillarity)
    images = crop_images(images)
    images = np.array(images).transpose((0, 3, 1, 2))  # Ensure proper shape for PyTorch
    labels = np.array(labels)
    return images, labels


def create_dataloaders(images, labels, batch_size, test_size, random_state):
    """
    Create train and test dataloaders.

    Args:
        images (ndarray): Image data.
        labels (ndarray): Labels for the images.
        batch_size (int): Batch size.
        test_size (float): Test set ratio.
        random_state (int): Random seed for splitting.

    Returns:
        train_loader, test_loader: DataLoader objects for training and testing.
    """
    # Define your transform pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Lambda(lambda x: x.to(torch.float32)),  # Enforce float32 type
        # transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        # transforms.Normalize(mean=0.5, std=0.2)
        transforms.Lambda(lambda x: x / 65535.0),
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # Permute axes from HWC to CHW (channels, height, width)
    ])
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=random_state)
    train_dataset = MultispectralDataset(X_train, y_train, transform=transform)
    test_dataset = MultispectralDataset(X_test, y_test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def initialize_model(images, labels, device):
    """
    Initialize the CNN model.

    Args:
        images (ndarray): Image data for shape testing.
        labels (ndarray): Labels for output size.
        device: Computation device.

    Returns:
        model: Initialized CNN model.
    """
    test_input = torch.zeros(images[0:1].shape)  # Create test input to calculate flattened size
    model = CNNModel.from_test_input(output_size=labels[0].size, test_input=test_input).to(device)
    return model


def validate_model(model, test_loader, device):
    """
    Validate the model and compute MSE.

    Args:
        model: The CNN model.
        test_loader: DataLoader for testing data.
        device: Computation device.

    Returns:
        Mean Squared Error for the test data.
    """
    model.eval()
    y_pred, y_true = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            y_pred.extend(outputs.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    mse = mean_squared_error(np.array(y_true), np.array(y_pred))
    return mse


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
    }
    # Save to file
    torch.save(metadata, file_path)
    print(f"Model and metadata saved to {file_path}")


if __name__ == "__main__":
    # Example paths and DataFrame
    image_folder1 = r"C:\Users\HEW\Projekte\5gla-ai-spectral-unmixing\data\registered"
    image_folder2 = r"C:\Users\HEW\Projekte\5gla-ai-spectral-unmixing\data\unmixing"
    df = read_soil_moisture()

    # val_mse = train_torch(image_folder=image_folder1, image_folder2=image_folder2, df=df,
    #                       batch_size=16, num_epochs=100, learning_rate=0.0005, test_size=0.2,
    #                       assumed_capillarity=0.1, verbose=True)

    # experiment to determine capillarity, best: 0.17
    for assumed_capillarity in np.linspace(0.01, 0.5, 10):
        abc = []
        for i in range (5):
            val_mse = train_torch(image_folder=image_folder1, image_folder2=image_folder2, df=df,
                        batch_size=32, num_epochs=20, learning_rate=0.0005, test_size=0.2, assumed_capillarity=assumed_capillarity)
            abc.append(val_mse)
        print(f"{assumed_capillarity} : {sum(abc) / len(abc)}")