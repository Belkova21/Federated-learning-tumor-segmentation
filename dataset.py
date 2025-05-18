import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class TumorDataset(Dataset):
    """
        Custom PyTorch Dataset for loading paired MRI slices and segmentation masks.

        Parameters:
            image_dir (str): Directory containing input images (.png).
            mask_dir (str): Directory containing segmentation masks (_mask.png).
            transform (dict): Dictionary of augmentation functions.
            debug (bool): If True, prints debug messages on errors.
    """
    def __init__(self, image_dir, mask_dir, transform=None, debug=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.debug = debug
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

        # Standard preprocessing: convert to grayscale and resize
        self.default_transformation = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((512, 512))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        try:
            # Load paths
            image_path = os.path.join(self.image_dir, self.images[index])
            name = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = os.path.join(self.mask_dir, f"{name}_mask.png")

            # Load images
            image = Image.open(image_path).convert('L')  # Ensure grayscale
            mask = Image.open(mask_path).convert('L')    # Mask is also grayscale

            # Apply default preprocessing
            image = self.default_transformation(image)
            mask = self.default_transformation(mask)

            # Apply random transformations (augmentations)
            if self.transform:
                image, mask = self._random_transform(image, mask)

            # Convert to tensors
            image = TF.to_tensor(image)  # [0, 1]
            mask = TF.to_tensor(mask)

            # Ensure mask is binary (0 or 1)
            mask = (mask > 0.5).float()

            sample = {
                'index': index,
                'image': image,
                'mask': mask
            }
            return sample

        except Exception as e:
            if self.debug:
                print(f"[ERROR] Failed to load sample at index {index}: {e}")
            raise e

    def _random_transform(self, image, mask):
        """
            Applies random data augmentations from the specified dictionary.
        """
        keys = list(self.transform.keys())
        random.shuffle(keys)

        for key in keys:
            if random.random() < 0.5:
                if self.debug:
                    print(f"[DEBUG] Applying {key}")
                if key == 'rotate':
                    angle = random.randint(15, 75)
                    image = self.transform[key](image, angle)
                    mask = self.transform[key](mask, angle)
                else:
                    image = self.transform[key](image)
                    mask = self.transform[key](mask)
        return image, mask


# ------------------------------------------------------
# Centralized Data Loading Function (used in CL training)
# ------------------------------------------------------

def single_data_load():
    custom_transform = {
        'hflip': TF.hflip,
        'vflip': TF.vflip,
        'rotate': TF.rotate
    }

    images_dir = './dataset/images'
    masks_dir = './dataset/masks'

    # Split sizes
    imege_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    total_files = len(imege_files)
    print(f"total_files : {total_files}")

    test_size = 100
    remaining = total_files - test_size
    train_size = int(0.8 * remaining)
    val_size = remaining - train_size

    print(f"Train: {train_size}, Validation: {val_size}, Test: {test_size}")

    # Load full dataset
    full_dataset = TumorDataset(image_dir=images_dir, mask_dir=masks_dir, transform=custom_transform)

    # Reproducible split
    torch.manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )


    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

# ------------------------------------------------------
# Federated Data Loading Function (used in FL scenario)
# ------------------------------------------------------

def double_data_load():
    images_dir = './dataset/images'
    masks_dir = './dataset/masks'
    custom_transform = {
        'hflip': TF.hflip,
        'vflip': TF.vflip,
        'rotate': TF.rotate
    }

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    total_files = len(image_files)
    test_size = 100
    remaining = total_files - test_size
    train_size = int(0.8 * remaining)
    val_size = remaining - train_size

    print(f"Total: {total_files} | Train: {train_size}, Val: {val_size}, Test: {test_size}")

    full_dataset = TumorDataset(image_dir=images_dir, mask_dir=masks_dir, transform=custom_transform)
    torch.manual_seed(42)  # For replication

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Split into two parts for two clients
    train_indices_1 = list(range(0, train_size // 2))
    train_indices_2 = list(range(train_size // 2, train_size))

    val_indices_1 = list(range(0, val_size // 2))
    val_indices_2 = list(range(val_size // 2, val_size))

    train_dataset_1 = Subset(train_dataset, train_indices_1)
    train_dataset_2 = Subset(train_dataset, train_indices_2)

    val_dataset_1 = Subset(val_dataset, val_indices_1)
    val_dataset_2 = Subset(val_dataset, val_indices_2)

    train_loader_1 = DataLoader(train_dataset_1, batch_size=8, shuffle=True)
    val_loader_1 = DataLoader(val_dataset_1, batch_size=8, shuffle=False)

    train_loader_2 = DataLoader(train_dataset_2, batch_size=8, shuffle=True)
    val_loader_2 = DataLoader(val_dataset_2, batch_size=8, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Hospital 1 - Train: {len(train_dataset_1)}, Val: {len(val_dataset_1)}")
    print(f"Hospital 2 - Train: {len(train_dataset_2)}, Val: {len(val_dataset_2)}")
    print(f"Test: {len(test_dataset)}")
    return train_loader_1,val_loader_1,train_loader_2,val_loader_2,test_loader
