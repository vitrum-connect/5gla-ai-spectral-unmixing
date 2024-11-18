import os

import cv2
import rasterio
from spectral import unmix
from sklearn.cluster import KMeans
import numpy as np
from app.modules.utils import reconstruct_image_with_removed_endmembers, plot


def find_best_n_elbow(data, max_clusters=5):
    # Store the sum of squared distances (inertia) for each value of n
    inertia = []
    cluster_nrs = range(1, max_clusters + 1)
    for n in cluster_nrs:
        kmeans = KMeans(n_clusters=n, random_state=0).fit(data)
        inertia.append(kmeans.inertia_)
    scores = [iner * nr_clusters for iner, nr_clusters in zip(inertia, cluster_nrs)]
    return cluster_nrs[scores.index(min(scores))]
    # Plot inertia vs. number of clusters
    # plt.plot(range(1, max_clusters + 1), inertia, marker='o')
    # plt.xlabel('Number of clusters (n)')
    # plt.ylabel('Inertia')
    # plt.title('Elbow Method for Optimal n')
    # plt.show()

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
    channels = []
    for file in files:
        with rasterio.open(os.path.join(file_path, file)) as src:
            # Read all bands (assuming multi-band image)
            channels.append(src.read())  # This will return a 3D NumPy array (bands, height, width)
    stacked_channels = np.concatenate(channels, axis=0)
    return stacked_channels

def calc_masks(channels, k_plant = 0.6, k_no_plant = 0.4):
    thresh_plant = np.max(channels[4]) * k_plant
    thresh_no_plant = np.max(channels[4]) * k_no_plant

    plant_mask = channels[4] > thresh_plant
    # plant_mask &= content_mask

    plot(plant_mask, "plant_mask")

    non_plant_mask = channels[4] < thresh_no_plant
    # non_plant_mask &= content_mask

    plot(non_plant_mask, "non_plant_mask")
    return plant_mask, non_plant_mask

def norm_channels(channels):
    # channels = channels[(2,1,0), :, :].transpose((1, 2, 0))
    channels = channels.transpose((1, 2, 0))
    channels = (channels / np.max(channels) * 255).astype(np.uint8)

    plot(channels[:,:,0:3], f"first_three_channels")
    return channels

def find_endmembers(channels, mask, n=None):
    slice = channels[mask]
    if n is None:
        n = find_best_n_elbow(slice)

    kmeans = KMeans(n_clusters=n, random_state=0).fit(slice)
    endmembers = kmeans.cluster_centers_
    return endmembers, n

def perform_unmix(channels, endmembers):
    unmix_result = unmix(channels, endmembers)
    # unmix_result = unmix_result.reshape((unmix_result.shape[-1], height, width))
    # unmix_result /= np.max(unmix_result)
    unmix_result[unmix_result < 0] = 0
    # sums = np.sum(unmix_result, axis=(2,))
    # for i in range(unmix_result.shape[-1]):
    #     unmix_result[:,:,i] /= sums
    # unmix_result *= -1 #BBB

    for i in range(unmix_result.shape[-1]):
        result = unmix_result[:, :, i]
        plot(result, f"result {i}", cmap='viridis')
    return endmembers, unmix_result

def reconstruct(channels, endmembers, unmix_result, remove_indices):
    reconstructed_image = reconstruct_image_with_removed_endmembers(unmix_result,
                                                                    endmembers,
                                                                    remove_indices=remove_indices,
                                                                    start_img=None,
                                                                    do_subtractive=False,
                                                                    # start_img=channels.astype(np.float64),
                                                                    # do_subtractive=True
                                                                    )
    reconstructed_image[reconstructed_image < 0] = 0
    reconstructed_image = (reconstructed_image / np.max(reconstructed_image) * 255).astype(np.uint8)
    return reconstructed_image
    # cv2.imwrite('../Reconstructed Image.png', reconstructed_image)
    # plot(reconstructed_image[:,:,0:3], "Reconstructed Image")

def run(file_path):
    channels = read_files(file_path)
    plant_mask, non_plant_mask = calc_masks(channels)
    channels = norm_channels(channels)

    endmembers_plant, n_plant = find_endmembers(channels, plant_mask)
    endmembers_non_plant, n_non_plant = find_endmembers(channels, non_plant_mask)
    print(n_plant, n_non_plant)
    endmembers = np.concatenate((endmembers_plant, endmembers_non_plant))

    endmembers, unmix_result = perform_unmix(channels, endmembers)
    remove_indices = list(range(n_plant, n_plant + n_non_plant))
    # remove_indices = list(range(0, n_plant))
    reconstructed = reconstruct(channels, endmembers, unmix_result, remove_indices)
    return reconstructed

if __name__ == "__main__":
    run()
