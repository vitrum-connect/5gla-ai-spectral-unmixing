import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

def plot_unmix_result(unmix_result):
    '''
    Plot the fractional abundances from unmixing result.

    ARGUMENTS:
        unmix_result: MxNxC array of fractional abundances for each endmember.
    '''

    # Number of endmembers (channels)
    num_endmembers = unmix_result.shape[2]

    # Plot each endmember's abundance
    fig, axes = plt.subplots(1, num_endmembers, figsize=(15, 5))

    for i in range(num_endmembers):
        abundance = unmix_result[:, :, i]

        # Plot the fractional abundances
        ax = axes[i]
        cax = ax.imshow(abundance, cmap='viridis', interpolation='nearest')
        ax.set_title(f'Endmember {i + 1}')
        plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def reconstruct_image_with_removed_endmembers(fractions, endmembers, remove_indices=(), start_img=None, do_subtractive=False):
    '''
    Reconstruct the image from the fractional abundances and endmember spectra,
    with specified endmembers removed from the reconstruction.

    ARGUMENTS:
        fractions: MxNxC array of endmember fractions
        endmembers: CxB array of endmember spectra (C endmembers, B spectral bands)
        remove_indices: List of endmember indices to remove (e.g., [0, 2] to remove the 1st and 3rd endmembers)

    RETURNS:
        reconstructed_image: MxNxB array representing the unmixed image with specific endmember contributions removed
    '''
    # Get dimensions
    height, width, num_endmembers = fractions.shape
    num_bands = endmembers.shape[1]

    # Copy the fractions array to avoid modifying the original
    modified_fractions = np.copy(fractions)

    # Set the fractions of the specified endmembers to zero
    for idx in remove_indices:
        modified_fractions[:, :, idx] = 0

    # Initialize the reconstructed image
    if start_img is None:
        start_img = np.zeros((height, width, num_bands))

    # Perform the reconstruction with the modified fractions
    for i in range(num_endmembers):
        for b in range(num_bands):
            # Add the contribution of each endmember to each band
            if do_subtractive:
                start_img[:, :, b] -= modified_fractions[:, :, i] * endmembers[i, b]
            else:
                start_img[:, :, b] += modified_fractions[:, :, i] * endmembers[i, b]

    return start_img

def plot(img, title, show=False, **kwargs):
    plt.imshow(img, **kwargs)#, cmap='viridis', interpolation='nearest')
    plt.title(title)
    if show:
        plt.show()
    else:
        plt.savefig(f"{title}.png", bbox_inches='tight')  # Save the entire plot with title
    plt.close()  # Close the figure to free up memory


def save_channels(img, output_dir='aligned_images', prefix='aligned_channel'):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    # Save each channel as a separate TIFF file
    channels = [img[:, :, i] for i in range(img.shape[2])] if len(img.shape) > 2 else [img]
    for i, channel in enumerate(channels):  # Assuming channels are in the last dimension
        output_file = os.path.join(output_dir, f"{prefix}_{i + 1}.tif")
        output_paths.append(output_file)
        cv2.imwrite(output_file, channel)
        print(f"Saved channel {i + 1} to {output_file}")
    return output_dir, output_paths
