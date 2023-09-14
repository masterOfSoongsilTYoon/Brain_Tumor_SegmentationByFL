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

from utils import calculate_accuracy, calculate_mIOU, ploting, save_csv

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parserer():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default='Baseline_Unet')
    parser.add_argument("--mode", type=str, default='unet', help="[unet, deeplab, mgunet]")
    parser.add_argument("--data", type=str, default='OCT', help="[OCT, brain]")
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=1e-3, type=float, metavar='N',
                        help='learning rate')
    parser.add_argument('-c', "--classification", default=False, type=bool, metavar='N',
                        help='classifier mode')
    parser.add_argument('-m', "--multimodal", default=False, type=bool, metavar='N',
                        help='classifier mode')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N',
                        help='mini-batch size (default: 8), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--backbone', default="101", type=str,
                        metavar='N',
                        help='[101, swin, efficient]')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')

    a = parser.parse_args()
    return a
def list_to_device(lis:list, DEVICE):
    result=[object.to(DEVICE) for object in lis]
    return result
def list_to_float(lis:list):
    result = [tensor.type(torch.float32)for tensor in lis]
    return result
def list_to_criterion(lis:list, criterion, label):
    result = [criterion(out.squeeze().type(torch.float32), value.squeeze().type(torch.float32)) for out, value in zip(lis, label) if not out is None]
    return lis_all_add(result)
def class_to_criterion(out, criterion,label):
    result = criterion(out.squeeze().type(torch.float32).to(DEVICE), label.type(torch.LongTensor).to(DEVICE))
    return result
def list_to_accuracy(tensor:list, function, label, type= torch.int32):
    result = function(tensor.type(type), label.type(type))
    return sum(result)/len(result)

def lis_all_add(lis:list):
    result=0
    for v in lis:
        result+=v
    return v
def list_to_mIOU(lis:list, function, label, threshold:float, type= torch.float64):
    result = [function(out.type(type),value.squeeze().type(type),threshold) for out,value in zip(lis, label)if not out is None]
    length = len(result)
    result=lis_all_add(result)
    return result/length
def train(net, train_dataloader, criterion, optimizer, valid_dataloader, a ,central_mode=False, data="OCT", classifier= None, coptimizer=None,multimodal=False):
    history={"loss":[], "acc":[], "mIOU":[]}
    for epoch in range(a.epochs):
        if not multimodal:
            net.train()
        else:
            for n in net:
                n.train()
        if not classifier is None:
            classifier.train()
        step_loss_list=[]
        acc = 0
        iou = 0
        for i, sample in enumerate(train_dataloader):
            optimizer.zero_grad()
            out_sample={}
            out_sample['label'] = torch.stack([sa["label"] for sa in sample], dim=0)
            out_sample['image'] = torch.stack([sa["image"] for sa in sample], dim=0)
            
            if a.classification:
                out_sample["class"] = torch.Tensor([sa["class"][0] for sa in sample])
            
            if data =="OCT":
                out = net(out_sample["image"].type(torch.float32).to(DEVICE))
                out, out_sample['label'] = list_to_device([out, out_sample['label'].squeeze()])
                out = out.type(torch.float64)
                # out_sample["label"] = torch.where(out_sample["label"]>0.5, 1.0, 0.0)
                loss = criterion(out.squeeze(), out_sample['label'].type(torch.float64))
                loss.backward()
                optimizer.step()
                
                acc+= 0
                iou+= calculate_mIOU(out.squeeze(), out_sample["label"], 0.2)
                
            elif data=="brain":
                if a.classification:
                    if multimodal:
                        out = [classifier(out_sample["image"].type(torch.float32).to(DEVICE)), net[0](out_sample['image'][out_sample["class"]==0,:,:,:].type(torch.float32).to(DEVICE)), net[1](out_sample['image'][out_sample["class"]==1,:,:,:].type(torch.float32).to(DEVICE)), 
                            net[2](out_sample['image'][out_sample["class"]==2,:,:,:].type(torch.float32).to(DEVICE)), net[3](out_sample['image'][out_sample["class"]==3,:,:,:].type(torch.float32).to(DEVICE)), net[4](out_sample['image'][out_sample["class"]==4,:,:,:].type(torch.float32).to(DEVICE))]
                        kind,CS,DU,EZ,FG,HT, out_sample['label'] = list_to_device([*out, out_sample['label'].squeeze()], DEVICE)
                        out = list_to_float([kind.squeeze(), CS.squeeze(),DU.squeeze(),EZ.squeeze(),FG.squeeze(),HT.squeeze()])
                        label = [out_sample['class'].to(DEVICE),out_sample['label'][out_sample["class"]==0,:,:].to(DEVICE), out_sample['label'][out_sample["class"]==1,:,:].to(DEVICE),
                                out_sample['label'][out_sample["class"]==2,:,:].to(DEVICE), out_sample['label'][out_sample["class"]==3,:,:].to(DEVICE), out_sample['label'][out_sample["class"]==4,:,:].to(DEVICE)]
                        loss = list_to_criterion(out[1:], criterion, label[1:])+ class_to_criterion(torch.stack([out[0]],dim=1),torch.nn.CrossEntropyLoss().to(DEVICE),label[0])
                        loss.backward()
                        optimizer.step()
                        coptimizer.step()
                        acc+= calculate_accuracy(torch.argmax(kind,dim=-1).squeeze(), label[0], 0.2)
                        iou+= list_to_mIOU(out[1:], calculate_mIOU, label=label[1:],threshold=0.2)
                    else:
                        out = [classifier(out_sample["image"].type(torch.float32).to(DEVICE)), net(out_sample['image'].type(torch.float32).to(DEVICE))]   
                        kind, out, out_sample['label'] = list_to_device([*out, out_sample['label'].squeeze()], DEVICE)
                        loss = criterion(out.type(torch.float64).squeeze(), out_sample['label'].type(torch.float64)) + class_to_criterion(torch.stack([kind],dim=1),torch.nn.CrossEntropyLoss().to(DEVICE),out_sample['class'].to(DEVICE))
                        loss.backward()
                        optimizer.step()
                        coptimizer.step()
                        acc+= calculate_accuracy(torch.argmax(kind,dim=-1).squeeze(), out_sample['class'], 0.2)
                        iou+= calculate_mIOU(out.squeeze(), out_sample["label"], 0.2)
                else:
                    out = net(out_sample["image"].type(torch.float32).to(DEVICE))
                    out, out_sample['label'] = list_to_device([out, out_sample['label'].squeeze()], DEVICE)
                    out = out.type(torch.float64)
                    loss = criterion(out.squeeze(), out_sample['label'].type(torch.float64))
                    loss.backward()
                    optimizer.step()
                
                    acc+= 0
                    iou+= calculate_mIOU(out.squeeze(), out_sample["label"], 0.2)
                
                step_loss_list.append(loss.item())
           
        history['loss'].append(sum(step_loss_list)/len(step_loss_list))
        if a.classification:
            history['acc'].append((acc/(i+1)).item())
        else:
            history['acc'].append(0)
        history['mIOU'].append((iou/(i+1)).item())
        
        if central_mode:
            print(f"Train==> epoch: {epoch}, loss: {sum(step_loss_list)/len(step_loss_list)}, acc: {acc/(i+1)}, mIou: {(iou/(i+1)).item()}")
            history2 = eval(net,net.parameters(), valid_dataloader, criterion, central_mode, classifier, data=data, multimodal=multimodal)  
        if data =="OCT":
            torch.save(net.state_dict(), f"./models/{a.version}/net.pkl")     
        elif data =="brain":
            if a.classification:
                torch.save(classifier.state_dict(), f"./models/{a.version}/classifier.pkl")
            if not multimodal: 
                torch.save(net.state_dict(), f"./models/{a.version}/net.pkl")   
            else:
                l=["CS", "DU", "EZ", "FG", "HT"]
                for indx, n in enumerate(net):
                    torch.save(n.stae_dict(), f"./models/{a.version}/{l[indx]}_net.pkl")
    return history, history2        
            
def eval(net, parameters, valid_dataloader, criterion, central_mode, classifier=None, data="OCT", multimodal=False):
    history={"loss":[], "acc":[], "mIOU":[]}
    with torch.no_grad():
        net.eval()
        if not classifier is None:
            classifier.eval()
        step_loss_list=[]
        acc= 0
        iou= 0
        for i, sample in enumerate(valid_dataloader):
            out_sample={}
            out_sample={}
            out_sample['label'] = torch.stack([sa["label"] for sa in sample], dim=0)
            out_sample['image'] = torch.stack([sa["image"] for sa in sample], dim=0)
            out = net(out_sample["image"].type(torch.float32).to(DEVICE))
            if not classifier is None:
                out_sample["class"] = torch.Tensor([sa["class"][0] for sa in sample])
                
            if data == "OCT":
                
                out, out_sample['label'] = list_to_device([out, out_sample['label'].squeeze()])
                out = out.type(torch.float64)
                # out_sample["label"] = torch.where(out_sample["label"]>0.5, 1.0, 0.0)
                
                loss = criterion(out.squeeze(), out_sample['label'].type(torch.float64))
                
                acc+= calculate_accuracy(out.squeeze().type(torch.int64), out_sample["label"].type(torch.int64), 2)
                iou+= calculate_mIOU(out.squeeze(), out_sample["label"], 0.2)
            elif data =="brain":
                if not classifier is None:
                    if multimodal:
                        out = [classifier(out_sample["image"].type(torch.float32).to(DEVICE)), net(out_sample['image'][out_sample["class"]==0,:,:,:].type(torch.float32).to(DEVICE)), net(out_sample['image'][out_sample["class"]==1,:,:,:].type(torch.float32).to(DEVICE)), 
                            net(out_sample['image'][out_sample["class"]==2,:,:,:].type(torch.float32).to(DEVICE)), net(out_sample['image'][out_sample["class"]==3,:,:,:].type(torch.float32).to(DEVICE)), net(out_sample['image'][out_sample["class"]==4,:,:,:].type(torch.float32).to(DEVICE))]
                        kind,CS,DU,EZ,FG,HT, out_sample['label'] = list_to_device([*out, out_sample['label'].squeeze()], DEVICE)
                        out = list_to_float([kind.squeeze(), CS.squeeze(),DU.squeeze(),EZ.squeeze(),FG.squeeze(),HT.squeeze()])
                        label = [out_sample['class'].to(DEVICE),out_sample['label'][out_sample["class"]==0,:,:].to(DEVICE), out_sample['label'][out_sample["class"]==1,:,:].to(DEVICE),
                                out_sample['label'][out_sample["class"]==2,:,:].to(DEVICE), out_sample['label'][out_sample["class"]==3,:,:].to(DEVICE), out_sample['label'][out_sample["class"]==4,:,:].to(DEVICE)]
                        loss = list_to_criterion(out[1:], criterion, label[1:])+ class_to_criterion(torch.stack([out[0]],dim=1),torch.nn.CrossEntropyLoss().to(DEVICE),label[0])
                        acc+= calculate_accuracy(torch.argmax(kind,dim=-1).squeeze(), label[0], 0.2)
                        iou+= list_to_mIOU(out[1:], calculate_mIOU, label=label[1:],threshold=0.2)
                    else:
                        out = [classifier(out_sample["image"].type(torch.float32).to(DEVICE)), net(out_sample['image'].type(torch.float32).to(DEVICE))]   
                        kind, out, out_sample['label'] = list_to_device([*out, out_sample['label'].squeeze()], DEVICE)
                        loss = criterion(out.type(torch.float64).squeeze(), out_sample['label'].type(torch.float64)) + class_to_criterion(torch.stack([kind],dim=1),torch.nn.CrossEntropyLoss().to(DEVICE),out_sample['class'].to(DEVICE))
                        acc+= calculate_accuracy(torch.argmax(kind,dim=-1).squeeze(), out_sample['class'], 0.2)
                        iou+= calculate_mIOU(out.squeeze(), out_sample["label"], 0.2)
                else:
                    out = net(out_sample["image"].type(torch.float32).to(DEVICE))
                    out, out_sample['label'] = list_to_device([out, out_sample['label'].squeeze()],DEVICE=DEVICE)
                    out = out.type(torch.float64)
                    loss = criterion(out.squeeze(), out_sample['label'].type(torch.float64))
                    acc+= 0
                    iou+= calculate_mIOU(out.squeeze(), out_sample["label"], 0.2)
            step_loss_list.append(loss.item())
        history['loss'].append(sum(step_loss_list)/len(step_loss_list))
        if a.classification:
            history['acc'].append((acc/(i+1)).item())
        else:
            history['acc'].append(0)
        history['mIOU'].append((iou/(i+1)).item())
        if central_mode:
            print(f"valid==> loss: {sum(step_loss_list)/len(step_loss_list)}, acc: {acc/(i+1)}, mIOU: {(iou/(i+1)).item()}")   
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
    if a.data =="OCT":
        net = Baseline_net(1, a.mode, a.data, a.backbone).to(DEVICE)
        cnet = None
    elif a.data =="brain":
        if a.mode =="unet":
            if not a.multimodal:
                net = Baseline_net(3, a.mode, a.data, a.backbone).to(DEVICE)
            else:
                net = [Baseline_net(3, a.mode, a.data, a.backbone).to(DEVICE) for _ in range(5)]
        elif a.mode =="deeplab":
            if not a.multimodal:
                net = Baseline_net(3, a.mode, a.data, a.backbone).to(DEVICE)
            else:
                net = [Baseline_net(3, a.mode, a.data, a.backbone).to(DEVICE) for _ in range(5)]
        elif a.mode =="mgunet":
            if not a.multimodal:
                net = Baseline_net(3, a.mode, a.data, a.backbone).to(DEVICE)
            else:
                net = None
        
        if a.classification:
            cnet = BrainClassifier(3, a.backbone).to(DEVICE)
        else:
            cnet = None
    dataName = f"{a.data}_"
    df = pd.read_csv(f"./CSV/{dataName}train_central.csv")
    train_dataset = CustomDataset(df, transform=False, mode=a.data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= a.batch_size, shuffle=True, num_workers=0, collate_fn = lambda x:x)
    valid_df = pd.read_csv(f'./CSV/{dataName}valid_central.csv')
    valid_dataset = CustomDataset(valid_df, transform=False, mode=a.data)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=a.batch_size, shuffle=False, num_workers=0,collate_fn = lambda x: x)
    
    optimizer = torch.optim.SGD(net.parameters(), lr=a.lr)
    if a.classification:
        coptimizer = torch.optim.SGD(cnet.parameters(), lr=a.lr)
    else:
        coptimizer = None
    criterion = nn.BCELoss().to(DEVICE)
    
    history = train(net, train_dataloader, criterion, optimizer, valid_dataloader, a=a, central_mode=True, data=a.data, classifier=cnet, coptimizer=coptimizer, multimodal=a.multimodal)
    return history


if __name__ == '__main__':
    a = parserer()
    history1, history2 = main(a)
    save_csv(history1, 'train', a.version)
    save_csv(history2, 'valid', a.version)
    
    ploting(history1, name= "train_"+a.version)
    ploting(history1, "mIOU",  name= "train_"+a.version)
    ploting(history1, "acc",  name= "train_"+a.version)
    
    ploting(history2, name= "valid_"+ a.version)
    ploting(history2, "mIOU",  name= "valid_"+ a.version)
    ploting(history2, "acc",  name= "valid_"+ a.version)