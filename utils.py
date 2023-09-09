import torch
from torch.nn.functional import one_hot
from torchmetrics import JaccardIndex
def calculate_accuracy(predictions, targets, num_classes:int):
    """
    Calculates the accuracy of predictions given the target labels.
    Args:
        predictions (torch.Tensor): Predicted labels (logits).
        targets (torch.Tensor): True labels.
    Returns:
        float: Accuracy value between 0 and 1.
    """
    predictions=predictions.cpu()
    targets = targets.cpu()
    predictions = one_hot(predictions, num_classes)
    targets = one_hot(targets, num_classes)
    correct = (targets==predictions).sum(dim=[1,2,3])
    all_units = targets.size(2)*targets.size(3)
    accuracy = (correct/all_units).mean().item()
    return accuracy

def calculate_mIOU(predictions, targets, threshold:float):
    iouf=JaccardIndex(task="binary", threshold=threshold, num_classes=1, num_labels=1)
    return iouf(predictions.cpu(), targets.cpu())