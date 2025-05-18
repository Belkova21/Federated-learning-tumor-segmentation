import torch

def evaluate_model_on_loader(model, dataloader, device):
    """
    Evaluates the model on the given dataloader.
    Computes and prints average Dice, Precision, and Recall.
    """

    model.eval()
    dice_scores = []
    precision_scores = []
    recall_scores = []

    def dice(preds, targets, smooth=1e-6):
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        inter = (preds_flat * targets_flat).sum()
        return (2 * inter + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)

    def precision(preds, targets, smooth=1e-6):
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        tp = (preds_flat * targets_flat).sum()
        return (tp + smooth) / (preds_flat.sum() + smooth)

    def recall(preds, targets, smooth=1e-6):
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        tp = (preds_flat * targets_flat).sum()
        return (tp + smooth) / (targets_flat.sum() + smooth)

    with torch.no_grad():
        for batch in dataloader:
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)

            output = model(image)
            dice_scores.append(dice(output, mask).item())
            precision_scores.append(precision(output, mask).item())
            recall_scores.append(recall(output, mask).item())

    avg_dice = sum(dice_scores) / len(dice_scores)
    avg_prec = sum(precision_scores) / len(precision_scores)
    avg_rec = sum(recall_scores) / len(recall_scores)

    print("\n--- AVERAGE RESULTS ---")
    print(f"Avg Dice:      {avg_dice:.4f}")
    print(f"Avg Precision: {avg_prec:.4f}")
    print(f"Avg Recall:    {avg_rec:.4f}")

    return avg_dice, avg_prec, avg_rec


import numpy as np
import torch


def evaluate_two_masks(pred_mask, gt_mask, threshold=0.5, smooth=1e-6):
    """
    Evaluates Dice, Precision, and Recall between two masks.

    Parameters:
        pred_mask: predicted mask (NumPy array or torch.Tensor)
        gt_mask: ground truth mask (NumPy array or torch.Tensor)
        threshold: threshold to binarize predictions
        smooth: smoothing factor to avoid division by zero

    Returns:
        (dice, precision, recall): float values of each metric
    """

    # Convert to tensors if necessary
    if isinstance(pred_mask, np.ndarray):
        pred_mask = torch.from_numpy(pred_mask)
    if isinstance(gt_mask, np.ndarray):
        gt_mask = torch.from_numpy(gt_mask)

    # Ensure float type
    pred_mask = pred_mask.float()
    gt_mask = gt_mask.float()

    # Threshold predicted mask
    pred_bin = (pred_mask > threshold).float()

    # Flatten both
    pred_flat = pred_bin.view(-1)
    gt_flat = gt_mask.view(-1)

    # True positives
    tp = (pred_flat * gt_flat).sum()
    precision = (tp + smooth) / (pred_flat.sum() + smooth)
    recall = (tp + smooth) / (gt_flat.sum() + smooth)
    dice = (2 * tp + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth)

    return dice.item(), precision.item(), recall.item()


