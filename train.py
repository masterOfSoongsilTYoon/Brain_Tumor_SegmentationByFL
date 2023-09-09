from dataset import CustomDataset
from networks import *
import numpy as np
import pandas as pd
import os
import argparse

import torch
import torch.utils.data
import torch.utils.data.distributed
import warnings
import math
from utils import calculate_accuracy, calculate_mIOU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parserer():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default='Baseline_Unet')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=1e-3, type=float, metavar='N',
                        help='learning rate')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N',
                        help='mini-batch size (default: 8), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')

    a = parser.parse_args()
    return a
def list_to_device(lis:list):
    result=[object.to(DEVICE) for object in lis]
    return result

def train(net, train_dataloader, criterion, optimizer, valid_dataloader, a ,central_mode=False):
    history={"loss":[], "acc":[], "mIOU":[]}
    for epoch in range(a.epochs):
        net.train()
        step_loss_list=[]
        acc = 0
        iou = 0
        for i, sample in enumerate(train_dataloader):
            optimizer.zero_grad()
            out_sample={}
            out_sample['label'] = torch.stack([sa["label"] for sa in sample], dim=0)
            out_sample['image'] = torch.stack([sa["image"] for sa in sample], dim=0)
            out = net(out_sample["image"].type(torch.float32).to(DEVICE))
            out, out_sample['label'] = list_to_device([out, out_sample['label'].squeeze()])
            out = out.type(torch.float64)
            out_sample["label"] = torch.where(out_sample["label"]>0.5, 1.0, 0.0)
            loss = criterion(out.squeeze(), out_sample['label'].type(torch.float64))
            loss.backward()
            optimizer.step()
            
            out= torch.where(out>0.5, 1.0, 0.0)
            
            acc+= calculate_accuracy(out.squeeze().type(torch.int64), out_sample["label"].type(torch.int64), 2)
            iou+= calculate_mIOU(out.squeeze(), out_sample["label"], 0.5)
            step_loss_list.append(loss.item())
            
        history['loss'].append(sum(step_loss_list)/len(step_loss_list))
        history['acc'].append(acc/(i+1))
        history['mIOU'].append(iou/(i+1))
        if central_mode:
            print(f"Train==> epoch: {epoch}, loss: {sum(step_loss_list)/len(step_loss_list)}, acc: {acc/(i+1)}, mIou: {iou/(i+1)}")
            history2 = eval(net,net.parameters(), valid_dataloader, criterion, central_mode,{})  
        torch.save(net.state_dict(), f"./models/{a.version}/Unet.pkl")      
    return history, history2        
            
def eval(net, parameters, valid_dataloader, criterion, central_mode, config):
    history={"loss":[], "acc":[], "mIOU":[]}
    with torch.no_grad():
        net.eval()
        step_loss_list=[]
        acc= 0
        iou= 0
        for i, sample in enumerate(valid_dataloader):
            out_sample={}
            out_sample={}
            out_sample['label'] = torch.stack([sa["label"] for sa in sample], dim=0)
            out_sample['image'] = torch.stack([sa["image"] for sa in sample], dim=0)
            out = net(out_sample["image"].type(torch.float32).to(DEVICE))
            out, out_sample['label'] = list_to_device([out, out_sample['label'].squeeze()])
            out = out.type(torch.float64)
            out_sample["label"] = torch.where(out_sample["label"]>0.5, 1.0, 0.0)
            
            loss = criterion(out.squeeze(), out_sample['label'].type(torch.float64))
            out= torch.where(out>0.5, 1.0, 0.0)
            acc+= calculate_accuracy(out.squeeze().type(torch.int64), out_sample["label"].type(torch.int64), 2)
            iou+= calculate_mIOU(out.squeeze(), out_sample["label"], 0.5)
            step_loss_list.append(loss.item())
        history['loss'].append(sum(step_loss_list)/len(step_loss_list))
        history['acc'].append(acc/(i+1))
        history['mIOU'].append(iou/(i+1))
        if central_mode:
            print(f"valid==> loss: {sum(step_loss_list)/len(step_loss_list)}, acc: {acc/(i+1)}, mIOU: {iou/(i+1)}")   
    return history

def main(a):
    warnings.filterwarnings(action='ignore')
    """ The main function for model training. """
    if os.path.exists('models') is False:
        os.makedirs('models')

    save_path = 'models/' + a.version
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    print(DEVICE, " 사용중")
    
    net = Baseline_Unet().to(DEVICE)
    
    df = pd.read_csv("./CSV/train_central.csv")
    train_dataset = CustomDataset(df, transform=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= a.batch_size, shuffle=True, num_workers=0, collate_fn = lambda x:x)
    valid_df = pd.read_csv('./CSV/valid_central.csv')
    valid_dataset = CustomDataset(valid_df, transform=False)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=a.batch_size, shuffle=False, num_workers=0,collate_fn = lambda x: x)
    
    optimizer = torch.optim.SGD(net.parameters(), lr=a.lr)
    criterion = nn.BCELoss().to(DEVICE)
    
    history = train(net, train_dataloader, criterion, optimizer, valid_dataloader, a=a, central_mode=True)
    return history

if __name__ == '__main__':
    a = parserer()
    history1, history2 = main(a)