import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from skimage.transform import resize

def visualize_oblique_projection(volume, save_path, angles=(90, 30), gt_img=None):
    if volume.ndim != 3:
        raise ValueError("Input volume must be a 3D numpy array.")
    
    # Rotate the volume for oblique projection
    rotated_volume = rotate(volume, angle=angles[0], axes=(1, 2), reshape=False, order=1)
    rotated_volume = rotate(rotated_volume, angle=angles[1], axes=(0, 2), reshape=False, order=1)
    
    # Max projection along an axis to get a 2D image
    image = np.max(rotated_volume, axis=0)
    image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
    
    # Plot side-by-side using matplotlib
    fig, ax = plt.subplots(1, 2 if gt_img is not None else 1, figsize=(8, 4))

    if gt_img is not None:
        gt_img = resize(gt_img, (image.shape[0], image.shape[1]), anti_aliasing=True)
        gt_img = (gt_img - gt_img.min()) / (gt_img.max() - gt_img.min())  # Normalize to [0, 1]
        ax[0].imshow(gt_img)
        ax[0].axis('off')
        ax[0].set_title("Ground Truth")
        
        # Plot the generated grayscale image next to it
        ax[1].imshow(image, cmap="viridis")
        ax[1].axis('off')
        ax[1].set_title("Generated Projection")
    else:
        # Plot only the generated grayscale image
        ax.imshow(image, cmap="viridis")
        ax.axis('off')

    # Save the plot to the given path
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

