import itertools
import os
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt
from tifffile import tifffile

from app.modules.persistent_storage_integration_service import PersistentStorageIntegrationService
from app.paths_handler import PathsManager

# files = psis._list_files_in_bucket(psis.bucket_name_for_stationary_images)
# for file in files:
#     capture_date = psis.get_time_of_capture(psis.bucket_name_for_stationary_images, file.object_name)
#     print(capture_date)
#     break
# Read CSV into a DataFrame

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


# capture_date = datetime.strptime(capture_date, "%Y:%m:%d %H:%M:%S")  # Adjust format as needed
# for date in df['Date Time']:
#     # Find the row with the minimum time difference
#     closest_row = df.loc[(date - capture_date).abs().idxmin()]



from datetime import datetime, timedelta

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

def donwload_closest_all(df):
    # Example usage
    dates_to_objects = get_dates_to_objects(psis, "cluster-odm")
    for ref_date in df['Date Time']:
        download_closest_image(psis, "cluster-odm", ref_date, dates_to_objects)


def train():
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, Flatten, Dense
    from tifffile import imread

    # Load the data
    df = pd.read_csv(r'../../data/Bodenfeuchte/Ostfalia011.csv')
    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%Y/%m/%d %H:%M:%S')

    # Remove rows with negative values in any "A.." columns
    a_columns = [col for col in df.columns if col.startswith('A')]
    df = df[(df[a_columns] >= 0).all(axis=1)]

    # Define a function to extract capture times and match to data
    def load_images_and_labels(image_folder, df):
        images = []
        labels = []

        for file_name in os.listdir(image_folder):
            if file_name.endswith('.tif'):
                file_path = os.path.join(image_folder, file_name)
                image = imread(file_path)

                # Metadata is often stored in the TIFF tags
                tags = image.pages[0].tags
                time_of_capture = tags.get('DateTime', None)
                time_of_capture = time_of_capture.value if time_of_capture else None


                # Extract capture time from metadata
                capture_time = ...  # Implement metadata extraction logic here

                # Match capture time to df row
                matched_row = df[df['Date Time'] == capture_time]
                if not matched_row.empty:
                    images.append(image)
                    labels.append(matched_row[a_columns].values[0])

        return images, labels

    # Load images and labels
    image_folder = "../../data/images"
    images, labels = load_images_and_labels(image_folder, df)

    # Convert to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Define a CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(images.shape[1], images.shape[2], images.shape[3])),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(a_columns), activation='linear')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16)

    # Plot the training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def train_torch():
    import pandas as pd
    import matplotlib.pyplot as plt
    import itertools
    import os
    from sklearn.model_selection import train_test_split
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from tifffile import imread
    import numpy as np

    # Load the data
    df = read_soil_moisture()

    # Define a custom dataset class
    class MultispectralDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]
            return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    # Define a function to extract capture times and match to data
    def load_images_and_labels(image_folder, df):
        images = []
        labels = []
        a_columns = [col for col in df.columns if col.startswith('A')]

        for file_name in os.listdir(image_folder):
            if file_name.endswith('.tif'):
                file_path = os.path.join(image_folder, file_name)
                image = imread(file_path)
                with tifffile.TiffFile(file_path) as tif:
                    # Metadata is often stored in the TIFF tags
                    tags = tif.pages[0].tags
                    capture_time = tags.get('DateTime', None)
                    capture_time = capture_time.value if capture_time else None

                # Ensure capture_time is a datetime object for comparison
                capture_time = pd.to_datetime(capture_time)
                # Calculate the absolute difference between capture_time and 'Date Time' in df
                df['Time Difference'] = (df['Date Time'] - capture_time).abs()
                # Find the row with the minimum time difference
                matched_row = df.loc[df['Time Difference'].idxmin()]

                # matched_row = df[df['Date Time'] == capture_time]
                if not matched_row.empty:
                    images.append(image)
                    labels.append(matched_row[a_columns].values[0])

        return images, labels

    # Load images and labels
    image_folder = r"C:\Users\HEW\Projekte\5gla-ai-spectral-unmixing\data\registered_test_imgs"
    images, labels = load_images_and_labels(image_folder, df)

    # Convert to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Create datasets and dataloaders
    train_dataset = MultispectralDataset(X_train, y_train)
    test_dataset = MultispectralDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Define a CNN model
    class CNNModel(nn.Module):
        def __init__(self, output_size):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv2d(5, 32, kernel_size=3)  # Assuming 5 channels
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.fc1 = nn.Linear(64 * 22 * 22, 128)  # Adjust dimensions based on input size and conv layers
            self.fc2 = nn.Linear(128, output_size)
            self.relu = nn.ReLU()

        def forward(self, X):
            X = self.relu(self.conv1(X))
            X = self.relu(self.conv2(X))
            X = X.view(X.size(0), -1)  # Flatten the tensor for the fully connected layers
            X = self.relu(self.fc1(X))
            X = self.fc2(X)
            return X


if __name__ == "__main__":
    # psis = PersistentStorageIntegrationService()
    train_torch()
    # df = read_soil_moisture()
    # create_plot(df)0