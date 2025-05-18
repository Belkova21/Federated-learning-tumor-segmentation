import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from time import time
#
class DynamicUNet(nn.Module):
    """
    DynamicUNet implements a configurable 5-level deep U-Net architecture for image segmentation tasks.

    The architecture follows the classical encoder-decoder design with skip connections, as introduced by Ronneberger et al. (2015).
    Each block consists of 3x3 convolutions with ReLU activations and optional max-pooling or transposed convolutions for downsampling/upsampling.

    Parameters:
        filters (list of int): Number of filters in each level of the encoder/decoder. Must contain exactly 5 values.
        input_channels (int): Number of input image channels (default: 1).
        output_channels (int): Number of output channels (e.g., 1 for binary mask prediction).

    Based on the original U-Net paper:
    Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", arXiv:1505.04597
    Adapted from: https://github.com/nasim-aust/Brain-Tumor-Segmentation-using-Unet-Model-in-2D-images
    """

    def __init__(self, filters, input_channels=1, output_channels=1):
        super(DynamicUNet, self).__init__()

        if len(filters) != 5:
            raise Exception(f"Filter list size {len(filters)}, expected 5!")

        padding = 1
        ks = 3

        # Encoding Part
        self.conv1_1 = nn.Conv2d(input_channels, filters[0], kernel_size=ks, padding=padding)
        self.conv1_2 = nn.Conv2d(filters[0], filters[0], kernel_size=ks, padding=padding)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(filters[0], filters[1], kernel_size=ks, padding=padding)
        self.conv2_2 = nn.Conv2d(filters[1], filters[1], kernel_size=ks, padding=padding)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(filters[1], filters[2], kernel_size=ks, padding=padding)
        self.conv3_2 = nn.Conv2d(filters[2], filters[2], kernel_size=ks, padding=padding)
        self.maxpool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(filters[2], filters[3], kernel_size=ks, padding=padding)
        self.conv4_2 = nn.Conv2d(filters[3], filters[3], kernel_size=ks, padding=padding)
        self.maxpool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.conv5_1 = nn.Conv2d(filters[3], filters[4], kernel_size=ks, padding=padding)
        self.conv5_2 = nn.Conv2d(filters[4], filters[4], kernel_size=ks, padding=padding)
        self.conv5_t = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)

        # Decoding Part
        self.conv6_1 = nn.Conv2d(filters[4], filters[3], kernel_size=ks, padding=padding)
        self.conv6_2 = nn.Conv2d(filters[3], filters[3], kernel_size=ks, padding=padding)
        self.conv6_t = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)

        self.conv7_1 = nn.Conv2d(filters[3], filters[2], kernel_size=ks, padding=padding)
        self.conv7_2 = nn.Conv2d(filters[2], filters[2], kernel_size=ks, padding=padding)
        self.conv7_t = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)

        self.conv8_1 = nn.Conv2d(filters[2], filters[1], kernel_size=ks, padding=padding)
        self.conv8_2 = nn.Conv2d(filters[1], filters[1], kernel_size=ks, padding=padding)
        self.conv8_t = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)

        self.conv9_1 = nn.Conv2d(filters[1], filters[0], kernel_size=ks, padding=padding)
        self.conv9_2 = nn.Conv2d(filters[0], filters[0], kernel_size=ks, padding=padding)

        # Output
        self.conv10 = nn.Conv2d(filters[0], output_channels, kernel_size=ks, padding=padding)

    def forward(self, x):
        # Encoding
        conv1 = F.relu(self.conv1_1(x))
        conv1 = F.relu(self.conv1_2(conv1))
        pool1 = self.maxpool1(conv1)

        conv2 = F.relu(self.conv2_1(pool1))
        conv2 = F.relu(self.conv2_2(conv2))
        pool2 = self.maxpool2(conv2)

        conv3 = F.relu(self.conv3_1(pool2))
        conv3 = F.relu(self.conv3_2(conv3))
        pool3 = self.maxpool3(conv3)

        conv4 = F.relu(self.conv4_1(pool3))
        conv4 = F.relu(self.conv4_2(conv4))
        pool4 = self.maxpool4(conv4)

        # Bottleneck
        conv5 = F.relu(self.conv5_1(pool4))
        conv5 = F.relu(self.conv5_2(conv5))

        # Decoding
        up6 = torch.cat((self.conv5_t(conv5), conv4), dim=1)
        conv6 = F.relu(self.conv6_1(up6))
        conv6 = F.relu(self.conv6_2(conv6))

        up7 = torch.cat((self.conv6_t(conv6), conv3), dim=1)
        conv7 = F.relu(self.conv7_1(up7))
        conv7 = F.relu(self.conv7_2(conv7))

        up8 = torch.cat((self.conv7_t(conv7), conv2), dim=1)
        conv8 = F.relu(self.conv8_1(up8))
        conv8 = F.relu(self.conv8_2(conv8))

        up9 = torch.cat((self.conv8_t(conv8), conv1), dim=1)
        conv9 = F.relu(self.conv9_1(up9))
        conv9 = F.relu(self.conv9_2(conv9))

        # Output
        output = self.conv10(conv9)  # <-- NO SIGMOID

        return output

    def summary(self, input_size=(1, 512, 512), batch_size=-1, device='cuda'):
        return summary(self, input_size, batch_size, device)


class DiceLoss(nn.Module):
    """
    Computes the soft Dice loss between predicted and target segmentation masks.

    Dice loss is commonly used in medical image segmentation tasks where class imbalance is significant.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice

class BCEDiceLoss(nn.Module):
    """
    Combines Binary Cross-Entropy loss with Dice loss for improved segmentation performance.

    This hybrid loss encourages both pixel-wise accuracy and overall region overlap.
    """
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.dice_loss = DiceLoss()

    def forward(self, preds, targets):
        return F.binary_cross_entropy_with_logits(preds, targets) + self.dice_loss(preds, targets)

class BrainTumorClassifier:
    """
        Implements a full training and evaluation pipeline for brain tumor segmentation using a U-Net model.

        This class provides methods for training with optional validation, testing using Dice score,
        and single-image prediction. It integrates with TensorBoard for logging and supports saving/loading model weights.

        The implementation is adapted and extended from:
        https://github.com/nasim-aust/Brain-Tumor-Segmentation-using-Unet-Model-in-2D-images

        Key extensions include:
        - Modular support for different loss functions (Dice, BCE+Dice)
        - Validation loss tracking and model checkpointing
        - Flexible use on CPU/GPU devices
    """

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = BCEDiceLoss().to(device)
        self.log_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.tb_writer = SummaryWriter(log_dir=f'logs/{self.log_path}')

    def train(self, trainloader, valloader=None, epochs=20, mini_batch_log=None, save_best_path=None, plot_samples=None):

        optimizer = optim.Adam(self.model.parameters(), lr=0.002)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.85, patience=2, verbose=True)

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')

        print('Starting Training...')
        for epoch in range(epochs):
            start = time()
            print(f'\n--- Epoch {epoch + 1}/{epochs} ---')

            # Training phase
            train_loss = self._train_epoch(trainloader, optimizer, mini_batch_log)
            history['train_loss'].append(train_loss)
            self.tb_writer.add_scalar('Train Loss', train_loss, epoch)
            self.tb_writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

            # Validation phase (if available)
            if valloader is not None:
                val_loss = self._validate_epoch(valloader)
                history['val_loss'].append(val_loss)
                self.tb_writer.add_scalar('Validation Loss', val_loss, epoch)
                scheduler.step(val_loss)
                print(f'Val Loss: {val_loss:.6f}')

                if save_best_path and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(save_best_path)
                    print(f'Model saved at {save_best_path} (Val Loss: {best_val_loss:.6f})')

            else:
                scheduler.step(train_loss)

            print(f'Train Loss: {train_loss:.6f} | Time: {time() - start:.2f}s')

        return history

    def _train_epoch(self, loader, optimizer, mini_batch_log):
        self.model.train()
        running_loss = 0.0

        for batch_idx, data in enumerate(loader):
            try:
                images = data['image'].to(self.device)
                masks = data['mask'].to(self.device)
            except Exception as e:
                print(f"Skipping batch {batch_idx} due to error: {e}")
                continue

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if mini_batch_log and (batch_idx + 1) % mini_batch_log == 0:
                avg_loss = running_loss / (mini_batch_log * loader.batch_size)
                print(f'Batch {batch_idx + 1}: Avg Loss = {avg_loss:.6f}')
                running_loss = 0.0

        epoch_loss = running_loss / len(loader.dataset)
        return epoch_loss

    def _validate_epoch(self, loader):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                try:
                    images = data['image'].to(self.device)
                    masks = data['mask'].to(self.device)
                except Exception as e:
                    print(f"Skipping batch {batch_idx} during validation due to error: {e}")
                    continue

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                running_loss += loss.item()

        val_loss = running_loss / len(loader.dataset)
        return val_loss

    def test(self, testloader, threshold=0.5):
        self.model.eval()
        scores = []

        if testloader.batch_size != 1:
            raise ValueError("Test batch size must be 1.")

        with torch.no_grad():
            for data in testloader:
                try:
                    image = data['image'].to(self.device).unsqueeze(0)
                    mask = data['mask'].cpu().numpy()
                except Exception as e:
                    print(f"Skipping sample due to error: {e}")
                    continue

                output = self.model(image)
                pred = (torch.sigmoid(output) > threshold).cpu().numpy()

                pred = np.resize(pred, (1, 512, 512))
                mask = np.resize(mask, (1, 512, 512))

                dice = self._dice_coefficient(pred, mask)
                scores.append(dice)

        return np.mean(scores)

    def predict(self, data, threshold=0.5):
        self.model.eval()
        image = data['image'].to(self.device).unsqueeze(0)
        mask = data['mask'].cpu().numpy()

        with torch.no_grad():
            output = self.model(image)
            pred = (torch.sigmoid(output) > threshold).cpu().numpy()

        image_np = np.resize(data['image'].cpu().numpy(), (512, 512))
        mask_np = np.resize(mask, (512, 512))
        pred_np = np.resize(pred, (512, 512))
        score = self._dice_coefficient(pred_np, mask_np)

        return image_np, mask_np, pred_np, score

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    @staticmethod
    def _dice_coefficient(pred, target):
        smooth = 1.0
        intersection = np.sum(pred * target)
        return (2. * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)
