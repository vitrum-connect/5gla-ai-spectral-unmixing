import os

import rasterio
from spectral import unmix
from sklearn.cluster import KMeans
import numpy as np
from app.modules.utils import reconstruct_image_with_removed_endmembers, plot, save_channels


def find_best_n_elbow(data, max_clusters=5):
    # Store the sum of squared distances (inertia) for each value of n
    inertia = []
    cluster_nrs = range(1, max_clusters + 1)
    for n in cluster_nrs:
        kmeans = KMeans(n_clusters=n, random_state=0).fit(data)
        inertia.append(kmeans.inertia_)
    scores = [iner * np.sqrt(nr_clusters) for iner, nr_clusters in zip(inertia, cluster_nrs)]
    return cluster_nrs[scores.index(min(scores))]

def read_file(file_path = "../Orthofotos/20240829_15_Ortho.tif"):
    with rasterio.open(file_path) as src:
        # Read all bands (assuming multi-band image)
        channels = src.read()  # This will return a 3D NumPy array (bands, height, width)
    channels = channels[:, channels.shape[1] // 4 : channels.shape[1] // 2, channels.shape[2] // 4 : channels.shape[2] // 2]
    content_mask = channels[-1] > 0 # last channel is mask of actual content
    channels = channels[0:-1]
    return channels, content_mask

def read_files(file_path):
    files = os.listdir(file_path)
    sorted_files = sorted(files, key=lambda f: int(f.split("_")[-1].split(".")[0]))
    channels = []
    for file in sorted_files:
        with rasterio.open(os.path.join(file_path, file)) as src:
            # Read all bands (assuming multi-band image)
            channels.append(src.read())  # This will return a 3D NumPy array (bands, height, width)
    stacked_channels = np.concatenate(channels, axis=0)
    channels = stacked_channels.transpose((1, 2, 0))
    return channels

def calc_masks(channels, k_plant = 0.6, k_no_plant = 0.4):
    thresh_plant = np.percentile(channels[:,:,4].flatten(),80)
    thresh_no_plant = np.percentile(channels[:,:,4].flatten(),10)
    # thresh_plant = im_max * k_plant
    # thresh_no_plant = im_max * k_no_plant
    plant_mask = channels[:,:,4] > thresh_plant
    non_plant_mask = channels[:,:,4] < thresh_no_plant
    return plant_mask, non_plant_mask

# def norm_channels(channels):
#     channels = channels.transpose((1, 2, 0))
#     channels = (channels / np.max(channels) * 255).astype(np.uint8)
#     return channels

def find_endmembers(channels, mask, n=None):
    slice = channels[mask]
    if n is None:
        n = find_best_n_elbow(slice)

    kmeans = KMeans(n_clusters=n, random_state=0).fit(slice)
    endmembers = kmeans.cluster_centers_
    return endmembers, n

def perform_unmix(channels, endmembers, clip_negative=True):
    abundances = unmix(channels, endmembers)
    if clip_negative:
        abundances[abundances < 0] = 0
    return endmembers, abundances

def reconstruct(channels, endmembers, unmix_result, remove_indices):
    reconstructed_image = reconstruct_image_with_removed_endmembers(unmix_result,
                                                                    endmembers,
                                                                    remove_indices=remove_indices,
                                                                    start_img=None,
                                                                    do_subtractive=False,
                                                                    )
    reconstructed_image[reconstructed_image < 0] = 0
    reconstructed_image = (reconstructed_image / np.max(reconstructed_image) * 255).astype(np.uint8)
    return reconstructed_image

def run(file_path):
    channels = read_files(file_path)
    plant_mask, non_plant_mask = calc_masks(channels)
    # channels = norm_channels(channels)

    endmembers_plant, n_plant = find_endmembers(channels, plant_mask)
    endmembers_non_plant, n_non_plant = find_endmembers(channels, non_plant_mask)
    endmembers = np.concatenate((endmembers_plant, endmembers_non_plant))

    endmembers, abundances = perform_unmix(channels, endmembers)
    remove_indices = list(range(n_plant, n_plant + n_non_plant))
    reconstructed = reconstruct(channels, endmembers, abundances, remove_indices)
    save_channels(reconstructed, output_dir="unmixed", prefix="unmixed")
    save_channels(abundances, output_dir="endmembers", prefix="endmembers")

    calc_savi(abundances, channels, n_plant)

    return reconstructed, abundances


def calc_savi(abundances, channels, n_plant):
    # calc N matrix
    abundances_plant = abundances[:, :, 0:n_plant]
    # abundances_non_plant = abundances[:, :, n_plant::]
    abundances_plant_sum = np.sum(abundances_plant, axis=2)
    # abundances_non_plant_sum = np.sum(abundances_non_plant, axis=2)
    soil_correction = np.abs((abundances_plant_sum / np.max(abundances_plant_sum)) ** 0.1 - 1)
    soil_correction = (soil_correction - np.min(soil_correction)) / np.max(soil_correction)

    # import cv2
    # b = cv2.normalize(soil_correction, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    # cv2.imwrite("soil_correction.png", b)
    #
    nir = channels[:, :, 4]
    red = channels[:, :, 2]
    savi = (nir - red) / (nir + red + soil_correction) * (1 / (soil_correction + 0.001))


if __name__ == "__main__":
    run()
