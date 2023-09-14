import torch
from torch.nn.functional import one_hot
from torchmetrics.classification import BinaryJaccardIndex
import matplotlib.pyplot as plt
import pandas as pd
import cv2

def calculate_accuracy(predictions, targets,threshold:float):
    """
    Calculates the accuracy of predictions given the target labels.
    Args:
        predictions (torch.Tensor): Predicted labels (logits).
        targets (torch.Tensor): True labels.
    Returns:
        float: Accuracy value between 0 and 1.
    """
    predictions= predictions.cpu()
    targets = targets.cpu()
    correct=(predictions==targets).sum(dim=0)
    all_units = targets.size(0)
    # predictions=predictions.cpu()
    # targets = targets.cpu()
    # correct = (1-targets+predictions).sum(dim=[1,2,3])
    # all_units = targets.size(1)*targets.size(2)*targets.size(3)
    # accuracy = (all_units-correct).mean().item()
    return correct/all_units

def calculate_mIOU(predictions, targets, threshold:float):
    targets=torch.where(targets>0.1, 1.0,0.0)
    iouf=BinaryJaccardIndex(threshold=threshold)
    result = iouf(predictions.cpu(), targets.cpu())
    return torch.nan_to_num(result, 0.0)

def ploting(history: dict, mode="loss", name=""):
    if mode =="loss":
        plt.plot(history["loss"], label=mode, color='r')
    elif mode =="mIOU":
        plt.plot(history["mIOU"], label=mode, color="g")
    elif mode =="acc":
        plt.plot(history["acc"], label=mode, color="b")
    plt.legend()
    plt.xlabel("epochs")
    plt.savefig(f"./Plot/{mode}_{name}.png")
    print(f"{mode} Plot saving!")
    plt.cla()

def save_csv(history, mode="train", name=""):
    df = pd.DataFrame({"loss":history["loss"], "mIOU":history["mIOU"], "acc":history["acc"]})
    df.to_csv(f"./CSV/Result_{mode}_{name}.csv")
    

def boundarization(img):
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    o = img.copy()
    e = cv2.erode(img.copy(), k)
    result = cv2.subtract(o,e)
    return result.astype("uint8")