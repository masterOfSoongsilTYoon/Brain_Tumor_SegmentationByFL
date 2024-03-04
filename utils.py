from typing import Any
import torch
from torch.nn.functional import one_hot
from torchmetrics.classification import BinaryJaccardIndex
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import math
from torchmetrics.functional import dice
from torchmetrics.classification import BinaryJaccardIndex
from sklearn.metrics import f1_score, accuracy_score
class PCA:
    def __init__(self, out_sample_images) -> None:
        self.out_images = out_sample_images
        
    def __call__(self, target_size,*args: Any, **kwds: Any) -> Any:
        num = self.out_images.size(0)
        out = self.out_images.reshape((num,-1))
        _,_,V=torch.pca_lowrank(out, center=True, niter=3)
        zresult=torch.matmul(out.T, V.T[:, :5000])
        xapprox  = torch.matmul(V.T[:, :5000], zresult.T)
        result = xapprox.reshape((num, *target_size))
        return result

def calculate_accuracy(predictions, targets,threshold:float):
    """
    Calculates the accuracy of predictions given the target labels.
    Args:
        predictions (torch.Tensor): Predicted labels (logits).
        targets (torch.Tensor): True labels.
    Returns:
        float: Accuracy value between 0 and 1.
    """
    batch=predictions.shape[0]
    predictions = np.where(predictions>threshold, 1, 0)
    predictions = np.reshape(predictions, (batch,-1))
    targets = np.where(targets>threshold, 1,0)
    targets = np.reshape(targets,(batch,-1))
    
    return accuracy_score(targets.astype(np.int32), predictions.astype(np.int32))

def calculate_f1Score(predictions, targets,threshold:float):
    """
    Calculates the accuracy of predictions given the target labels.
    Args:
        predictions (torch.Tensor): Predicted labels (logits).
        targets (torch.Tensor): True labels.
    Returns:
        float: Accuracy value between 0 and 1.
    """
    batch=predictions.shape[0]
    predictions = np.where(predictions>threshold, 1, 0)
    predictions = np.reshape(predictions, (batch,-1))
    targets = np.where(targets>threshold, 1,0)
    targets = np.reshape(targets,(batch,-1))
    
    return f1_score(targets.astype(np.int32), predictions.astype(np.int32), average="weighted")

def calculate_mIOU(predictions, targets, threshold:float, DEVICE)->torch.Tensor:
    iouf = BinaryJaccardIndex()
    
    
    predictions = predictions.flatten()
    
    targets = targets.flatten()
    predictions=torch.where(predictions.cpu()>0.5, 1,0)
    targets = torch.where(targets.cpu()>0.5, 1,0)
    if torch.max(targets).item() ==0:
        if torch.max(predictions).item ==0:
            return 1
        else:
            return 0
    else:
        return iouf(predictions, targets)

def calculate_DiceCE(predictions, targets)->torch.Tensor:
    predictions = predictions.flatten()
    targets = targets.flatten()
    predictions=torch.where(predictions.cpu()>0.5, 1,0)
    targets = torch.where(targets.cpu()>0.5, 1,0)
    iouf = dice
   
    # if torch.max(targets).item() ==0:
    #     if torch.max(predictions).item ==0:
    #         return 1
    #     else:
    #         return 0
    return iouf(predictions, targets)
   
    

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
    df = pd.DataFrame(history)
    df.to_csv(f"./CSV/Result_{mode}_{name}.csv")
    

def boundarization(img):
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    o = img.copy()
    e = cv2.erode(img.copy(), k)
    result = cv2.subtract(o,e)
    return result.astype("uint8")

def calculate_ASD(img):
    _,img = cv2.threshold(img, 51.0, 255.0, cv2.THRESH_BINARY)
    edge = boundarization(img)
    contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    xlis = []
    ylis = []
    if contours is not None:
        for i in contours:
            M = cv2.moments(i, True)
            if M["m00"] != 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                xlis.append(cX)
                ylis.append(cY)
            else:
                pass
        if len(xlis)>0:
            meanX = sum(xlis)//len(xlis)
            meanY = sum(ylis)//len(ylis)   
            # vec=np.array([(x,y) for y in np.arange(256) for x in np.arange(256)], dtype="int32")
            # for cnt in contours:
            #     for vector in cnt:
            #         vector[0]
            radial_vec=np.zeros((256,256))
            index=np.where(img!=0)
            for y,x in zip(*index):
                radial_vec[y,x] =math.sqrt((x-meanX)**2+(y-meanY)**2)
            

            normalize_distance=(radial_vec-radial_vec.min())/(radial_vec.max()-radial_vec.min())
            mean_radial_distance=np.average(normalize_distance)
            std_radial_distance=np.sqrt(((normalize_distance-mean_radial_distance)**2))
        else:
            radial_vec= np.zeros((256,256))
            std_radial_distance = np.zeros((256,256))
    else:
        radial_vec= np.zeros((256,256))
        std_radial_distance = np.zeros((256,256))
    return {"std":std_radial_distance, "rd": radial_vec, "edge":edge}

def calculate_MF(img, ASD=False):
    if not ASD:
        return cv2.blur(calculate_ASD(img), (3,3))
    else:
        return cv2.blur(img,(3,3))