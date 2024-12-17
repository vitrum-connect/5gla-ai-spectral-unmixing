import numpy as np
import torch
from tifffile import imread

from app.modules.soil_prediction import crop_images
from app.modules.model import CNNModel


def load_model_and_infer(model_path, image_path1, image_path2, device=None):
    """
    Load a trained model and perform inference on two images.

    Args:
        model_path (str): Path to the trained model file.
        image_path1 (str): Path to the first image (e.g., registered folder).
        image_path2 (str): Path to the second image (e.g., unmixing folder).
        device (torch.device): Device to run inference on (CPU or GPU). Defaults to auto-detect.

    Returns:
        prediction (numpy array): Model's predicted output.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")



    # Load the trained model
    # Load metadata and model state
    metadata = torch.load(model_path, map_location=device)
    state_dict = metadata['state_dict']
    flattened_size = metadata['flattened_size']
    output_size = metadata['output_size']
    input_shape = metadata['input_shape']

    model = CNNModel(output_size=output_size, input_shape=input_shape, flattened_size=flattened_size)  # Assuming output size is 9 (adjust based on your labels)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load and preprocess the two images
    image1 = imread(image_path1)
    image2 = imread(image_path2)
    # Combine the images into a single input tensor (stack channels)
    image = np.dstack([image1, image2])
    image = image[0:input_shape[2], 0:input_shape[3]].transpose((2, 0, 1))
    input_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dim
    input_tensor = input_tensor.to(device)


    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    prediction = output.cpu().numpy()
    return prediction


# def preprocess_image(image_path):
#     """
#     Load an image and preprocess it for the model.
#
#     Args:
#         image_path (str): Path to the image file.
#
#     Returns:
#         torch.Tensor: Preprocessed image tensor of shape [1, C, H, W].
#     """
#     image = imread(image_path)  # Read image
#     if len(image.shape) == 2:  # Grayscale image, add channel dimension
#         image = np.expand_dims(image, axis=-1)
#     image = np.transpose(image, (2, 0, 1))  # Convert to [C, H, W]
#     image = crop_images([image])[0]  # Crop to the smallest dimensions if needed
#     image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dim
#     return image_tensor

if __name__ == "__main__":
    model_path = "multispectral_model.pth"
    image_path1 = r"C:\Users\HEW\Projekte\5gla-ai-spectral-unmixing\data\registered\Feldversuch KI-Use-Case\1410_1612-2110_1052\007\IMG_1400.tif"
    image_path2 = r"C:\Users\HEW\Projekte\5gla-ai-spectral-unmixing\data\unmixing\Feldversuch KI-Use-Case\1410_1612-2110_1052\007\IMG_1400_savi.tif"

    prediction = load_model_and_infer(model_path, image_path1, image_path2)
    print("Model Prediction:", prediction)

