import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageprocessing.micasense.imageutils as imageutils
import imageprocessing.micasense.capture as capture

def register_images(imagePath=os.path.join('../..', 'data'),
                    match_index=1,  # Index of the band, here we use green
                    max_alignment_iterations=30,
                    warp_mode=cv2.MOTION_HOMOGRAPHY,
                    pyramid_levels=None,  # for 10-band imagery we use a 3-level pyramid. In some cases
                    name_pattern='*.tif'
                    ):
    imageNames = glob.glob(os.path.join(imagePath, name_pattern))
    captured = capture.Capture.from_filelist(imageNames)

    img_type = 'reflectance' if captured.dls_present() else "radiance"

    print("Aligning images. Depending on settings this can take from a few seconds to many minutes")
    # Can potentially increase max_iterations for better results, but longer runtimes
    warp_matrices, alignment_pairs = imageutils.align_capture(captured,
                                                              ref_index=match_index,
                                                              max_iterations=max_alignment_iterations,
                                                              warp_mode=warp_mode,
                                                              pyramid_levels=pyramid_levels,
                                                              multithreaded=False)

    print("Finished Aligning, warp matrices={}".format(warp_matrices))

    im_aligned = captured.create_aligned_capture(warp_matrices=warp_matrices, motion_type=warp_mode, img_type=img_type)
    return captured, im_aligned


def save_aligned_channels(im_aligned, output_dir='aligned_images', prefix='aligned_channel'):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    # Save each channel as a separate TIFF file
    for i in range(im_aligned.shape[2]):  # Assuming channels are in the last dimension
        channel = im_aligned[:, :, i]

        # Normalize channel to [0, 65535] for 16-bit TIFF
        normalized_channel = cv2.normalize(channel, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)

        # Convert to 16-bit unsigned integer
        output_channel = normalized_channel.astype(np.uint16)

        # Save the channel
        output_file = os.path.join(output_dir, f"{prefix}_{i + 1}.tif")
        output_paths.append(output_file)
        cv2.imwrite(output_file, output_channel)
        print(f"Saved normalized channel {i + 1} to {output_file}")
    return output_dir, output_paths

def plot(captured, im_aligned):
    figsize = (16, 13)  # use this size for export-sized display

    rgb_band_indices = [captured.band_names().index('Red'), captured.band_names().index('Green'),
                        captured.band_names().index('Blue')]
    cir_band_indices = [captured.band_names().index('NIR'), captured.band_names().index('Red edge'),
                        captured.band_names().index('Red edge')]

    # Create a normalized stack for viewing
    im_display = np.zeros((im_aligned.shape[0], im_aligned.shape[1], im_aligned.shape[2]), dtype=np.float32)

    im_min = np.percentile(im_aligned[:, :, rgb_band_indices].flatten(), 0.5)  # modify these percentiles to adjust contrast
    im_max = np.percentile(im_aligned[:, :, rgb_band_indices].flatten(),
                           99.5)  # for many images, 0.5 and 99.5 are good values

    # # for rgb true color, we use the same min and max scaling across the 3 bands to
    # # maintain the "white balance" of the calibrated image
    for i in rgb_band_indices:
        im_display[:, :, i] = imageutils.normalize(im_aligned[:, :, i], im_min, im_max)
    rgb = im_display[:, :, rgb_band_indices]

    # for cir false color imagery, we normalize the NIR,R,G bands within themselves, which provides
    # the classical CIR rendering where plants are red and soil takes on a blue tint
    for i in cir_band_indices:
        im_display[:, :, i] = imageutils.normalize(im_aligned[:, :, i])
    cir = im_display[:, :, cir_band_indices]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].set_title("Red-Green-Blue Composite")
    axes[0].imshow(rgb)
    axes[1].set_title("Color Infrared (CIR) Composite")
    axes[1].imshow(cir)
    plt.show()

def run(image_path):
    captured, im_aligned = register_images(image_path)
    output_folder, output_paths = save_aligned_channels(im_aligned)
    return captured, im_aligned, output_folder


if __name__ == '__main__':
    captured, im_aligned = run()
    plot(captured, im_aligned)
