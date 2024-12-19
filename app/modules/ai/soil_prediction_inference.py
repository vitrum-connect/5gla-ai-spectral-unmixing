import numpy as np
import torch
from skimage.transform import resize
from tifffile import imread

from app.modules.ai.model import CNNModel
from app.modules.ai.soil_prediction_training import prepare_images


def load_model_and_infer_disk(image_path1, image_path2, device=None, model_path="multispectral_model.pth", ):
    # Load and preprocess the two images
    image1 = imread(image_path1)
    image2 = imread(image_path2)
    return load_model_and_infer(image1, image2, device, model_path)

def load_model_and_infer(image1, image2, device=None, model_path="multispectral_model.pth"):
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
    num_layers = metadata['num_layers']
    complexity = metadata['complexity']

    model = CNNModel(flattened_size, input_shape, output_size, complexity, num_layers)  # Assuming output size is 9 (adjust based on your labels)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Combine the images into a single input tensor (stack channels)
    image = np.dstack([image1, image2])

    image = crop_to_aspect_ratio(image, input_shape[3]/input_shape[2])
    image = resize(image, (input_shape[2], input_shape[3]))
    image = image.transpose((2, 0, 1))
    input_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dim
    input_tensor = input_tensor.to(device)


    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    prediction = output.cpu().numpy()
    return prediction

def crop_to_aspect_ratio(image, desired_aspect_ratio):
    """
    Crop an image to a desired aspect ratio while preserving the center.

    Args:
        image (numpy.ndarray): Input image as a 2D (grayscale) or 3D (color) array.
        desired_aspect_ratio (float): Desired aspect ratio (width / height).

    Returns:
        numpy.ndarray: Cropped image with the desired aspect ratio.
    """
    height, width = image.shape[:2]
    current_aspect_ratio = width / height

    if np.isclose(current_aspect_ratio, desired_aspect_ratio, atol=1e-3):
        # Aspect ratio is already close to desired, return the original image
        return image

    if current_aspect_ratio > desired_aspect_ratio:
        # Wider than desired, crop the width
        new_width = int(height * desired_aspect_ratio)
        start_x = (width - new_width) // 2
        end_x = start_x + new_width
        cropped_image = image[:, start_x:end_x]
    else:
        # Taller than desired, crop the height
        new_height = int(width / desired_aspect_ratio)
        start_y = (height - new_height) // 2
        end_y = start_y + new_height
        cropped_image = image[start_y:end_y, :]

    return cropped_image

if __name__ == "__main__":
    model_path = "ai\multispectral_model.pth"
    image_path1 = r"C:\Users\HEW\Projekte\5gla-ai-spectral-unmixing\data\registered\Feldversuch KI-Use-Case\1410_1612-2110_1052\007\IMG_1400.tif"
    image_path2 = r"C:\Users\HEW\Projekte\5gla-ai-spectral-unmixing\data\unmixing\Feldversuch KI-Use-Case\1410_1612-2110_1052\007\IMG_1400_savi.tif"

    prediction = load_model_and_infer_disk(image_path1, image_path2, model_path=model_path)
    print("Model Prediction:", prediction)

