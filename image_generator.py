import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def result_comparison(image, mask_gt, mask_cl, mask_fl, title="", transparency=0.4, save_path=None):
    """
    Visualizes and compares segmentation masks from CL and FL models with ground truth.

    Parameters:
        image (np.ndarray): Original grayscale MRI image (2D).
        mask_gt (np.ndarray): Ground truth binary mask (0/1 or 0/255).
        mask_cl (np.ndarray): CL model binary prediction mask.
        mask_fl (np.ndarray): FL model binary prediction mask.
        title (str): Title of the plot.
        transparency (float): Alpha for overlay masks on image.
        save_path (str): Path to save the image, if provided.

    Returns:
        None
    """

    # Normalize masks if needed (0/255 → 0/1)
    if mask_gt.max() > 1: mask_gt = mask_gt / 255
    if mask_cl.max() > 1: mask_cl = mask_cl / 255
    if mask_fl.max() > 1: mask_fl = mask_fl / 255

    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(title, fontsize=18)

    # Row 1: masks only
    axs[0, 0].imshow(mask_gt, cmap='gray')
    axs[0, 0].set_title("Ground Truth Mask")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(mask_cl, cmap='gray')
    axs[0, 1].set_title("CL Prediction Mask")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(mask_fl, cmap='gray')
    axs[0, 2].set_title("FL Prediction Mask")
    axs[0, 2].axis("off")

    # Row 2: Overlays on image
    axs[1, 0].imshow(image, cmap='gray')
    axs[1, 0].imshow(mask_gt, cmap='Reds', alpha=transparency)
    axs[1, 0].set_title("Ground Truth Overlay")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(image, cmap='gray')
    axs[1, 1].imshow(mask_cl, cmap='Greens', alpha=transparency)
    axs[1, 1].set_title("CL Overlay")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(image, cmap='gray')
    axs[1, 2].imshow(mask_fl, cmap='Blues', alpha=transparency)
    axs[1, 2].set_title("FL Overlay")
    axs[1, 2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Obrázok uložený do: {save_path}")

    plt.show()


mri_path = "./povodny_obrazok.png"
true_mask_path = "./originalna_maska.png"
cl_mask_path = "./CL_maska.png"
fl_mask_path = "./FL_maska.png"


# Load images in grayscale
mri_image = np.array(Image.open(mri_path).convert("L"))
true_mask = np.array(Image.open(true_mask_path).convert("L"))
cl_mask = np.array(Image.open(cl_mask_path).convert("L"))
fl_mask = np.array(Image.open(fl_mask_path).convert("L"))

result_comparison(
    image=mri_image,
    mask_gt=true_mask,
    mask_cl=cl_mask,
    mask_fl=fl_mask,
    title="Comparison of Segmentation Results",
    save_path="segmentation_comparison.png"
)