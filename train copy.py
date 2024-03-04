from dataset import CustomDataset
from networks import *
import numpy as np
import pandas as pd
import os
import argparse
import random
import torch
import torch.utils.data
import torch.utils.data.distributed
import warnings


from utils import calculate_accuracy, calculate_mIOU, save_csv, calculate_f1Score, calculate_DiceCE, DiceCELoss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parserer():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default='Baseline_Unet')
    parser.add_argument("--mode", type=str, default='unet', help="[unet, deeplab, mgunet]")
    parser.add_argument("--data", type=str, default='brain', help="[OCT, brain]")
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--num', default=1, type=int, metavar='N',
                        help='client number')
    parser.add_argument('--lr', default=1e-3, type=float, metavar='N',
                        help='learning rate')
    parser.add_argument('--merge', default=False, type=bool, metavar='N',
                        help='merge mode')
    parser.add_argument('--transfered', default=None, type=str, metavar='N',
                        help='transfered learning')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N',
                        help='mini-batch size (default: 8), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--backbone', default="101", type=str,
                        metavar='N',
                        help='[101, swin, efficient]')
    parser.add_argument('--seed', default=777, type=int,
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
def train(net, train_dataloader, criterion, optimizer, valid_dataloader, a ,central_mode=False, data="OCT"):
    history={"loss":[]}
    maxi=0
    if central_mode:
        history2={"loss":[], "acc":[], "mIOU":[], "f1Score": [], "dice":[]}
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
            # out_sample['std'] = torch.stack([sa["std"] for sa in sample], dim=0)
            
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
                
                # print(out_sample['std'].size())
                out = net({"x":out_sample["image"].type(torch.float32).to(DEVICE)})
                
                out, out_sample['label'] = list_to_device([out, out_sample['label']],DEVICE=DEVICE)
                out = out.type(torch.float64).to(DEVICE)
                out_sample["label"] = torch.where(out_sample["label"]>0.5, 1, 0)
                loss = criterion(out, out_sample['label'].type(torch.float64).to(DEVICE))
                
                # loss = calculate_DiceCELoss(out.squeeze(), out_sample["label"].squeeze())
                loss.backward()
                optimizer.step()
            
                # iou+= calculate_mIOU(out.squeeze(), out_sample["label"].squeeze(), 0.4, DEVICE)
                
                step_loss_list.append(loss.item())
           
        history['loss'].append(sum(step_loss_list)/len(step_loss_list))
        # history['mIOU'].append((iou/(i+1)).item())
        
        if central_mode:
            print(f"Train==> epoch: {epoch}, loss: {sum(step_loss_list)/len(step_loss_list)}")
            for key, value in eval(net, valid_dataloader, nn.BCEWithLogitsLoss().to(DEVICE), central_mode, data=data, a=a).items():
                history2[key].append(value[0])
                if key=="mIOU" and maxi<= value[0]:
                    torch.save(net.state_dict(), f"./models/{a.version}/net{a.num}.pkl")
                    maxi = value[0]        
        else:
            torch.save(net.state_dict(), f"./models/{a.version}/net{a.num}.pkl")
    if central_mode:
        return history, history2        
    else:
        return history
            
def eval(net, valid_dataloader, criterion, central_mode, data="OCT",  a=None):
    history={"loss":[], "acc":[], "mIOU":[], "f1Score":[], "dice":[]}
    with torch.no_grad():
        net.eval()
        step_loss_list=[]
        acc= 0
        iou= 0
        f1sc=0
        dices=0
        for i, sample in enumerate(valid_dataloader):
            out_sample={}
            out_sample['label'] = torch.stack([sa["label"] for sa in sample], dim=0)
            out_sample['image'] = torch.stack([sa["image"] for sa in sample], dim=0)
            # out_sample['std'] = torch.stack([sa["std"] for sa in sample], dim=0)
            
            if data == "OCT":
                out, out_sample['label'] = list_to_device([out, out_sample['label'].squeeze()])
                out = out.type(torch.float64)
                # out_sample["label"] = torch.where(out_sample["label"]>0.5, 1.0, 0.0)
                loss = criterion(out.squeeze(), out_sample['label'].type(torch.float64))
                acc+= calculate_accuracy(out.squeeze().type(torch.int64), out_sample["label"].type(torch.int64), 2)
                iou+= calculate_mIOU(out.squeeze(), out_sample["label"], 0.2)
            elif data =="brain":
                
                out = net({"x":out_sample["image"].type(torch.float32).to(DEVICE)})
                out, out_sample['label'] = list_to_device([out, out_sample['label']],DEVICE=DEVICE)
                out_sample["label"] = torch.where(out_sample["label"]>0.5, 1, 0)
                out = out.type(torch.float64).to(DEVICE)
                loss = criterion(out.squeeze(), out_sample['label'].type(torch.float64).to(DEVICE).squeeze())
                # loss = criterion(out.squeeze(), out_sample["label"].squeeze())
                iou+= calculate_mIOU(out.squeeze(), out_sample["label"].squeeze(), 0.4, DEVICE)
                acc+= calculate_accuracy(out_sample['label'].cpu().squeeze().detach().numpy(), out.cpu().squeeze().detach().numpy(),0.4)
                f1sc+= calculate_f1Score(out_sample['label'].cpu().squeeze().detach().numpy(), out.cpu().squeeze().detach().numpy(),0.4)
                dices+= calculate_DiceCE(out.squeeze(), out_sample["label"].squeeze())
            step_loss_list.append(loss.item())
        history['loss'].append(sum(step_loss_list)/len(step_loss_list))
        history['acc'].append(acc/(i+1))
        history['f1Score'].append(f1sc/(i+1))
        history['mIOU'].append((iou/(i+1)).item())
        history['dice'].append((dices/(i+1)).item())
        if central_mode:
            print(f"valid==> loss: {sum(step_loss_list)/len(step_loss_list)}, acc: {acc/(i+1):.4f}, f1Score: {f1sc/(i+1):.4f}, mIOU: {(iou/(i+1))}, Dice:{dices/(i+1)}")   
    return history

def main(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    torch.cuda.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    
    warnings.filterwarnings(action='ignore')
    """ The main function for model training. """
    if os.path.exists('models') is False:
        os.makedirs('models')

    save_path = 'models/' + a.version
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    print(DEVICE, " 사용중")
    if a.data =="OCT":
        net = Baseline_net(1, a.mode, a.data, a.backbone, a.merge).to(DEVICE)
        cnet = None
    elif a.data =="brain":
        if a.mode =="unet":
            net = Baseline_net(1, a.mode, a.data, a.backbone, a.merge).to(DEVICE)
        elif a.mode =="deeplab":
            net = Baseline_net(3, a.mode, a.data, a.backbone, a.merge).to(DEVICE)
        elif a.mode =="mgunet":
            net = Baseline_net(3, a.mode, a.data, a.backbone, a.merge).to(DEVICE)
    if a.transfered is not None:
        net.load_state_dict(torch.load(a.transfered))
    dataName = f"{a.data}_"
    df = pd.read_csv(f"./CSV/{dataName}train2_central.csv")
    train_dataset = CustomDataset(df, transform=False, mode=a.data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= a.batch_size, shuffle=True, num_workers=0, collate_fn = lambda x:x)
    valid_df = pd.read_csv(f'./CSV/{dataName}valid2_central.csv')
    valid_dataset = CustomDataset(valid_df, transform=False, mode=a.data)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=a.batch_size, shuffle=False, num_workers=0,collate_fn = lambda x: x)
    
    optimizer = torch.optim.SGD(net.parameters(), lr=a.lr)
    # criterion = DiceCELoss().to(device=DEVICE)
    criterion = nn.BCELoss().to(DEVICE)
    
    history1, history2 = train(net, train_dataloader, criterion, optimizer, valid_dataloader, a=a, central_mode=True, data=a.data)
    return history1, history2


if __name__ == '__main__':
    a = parserer()
    history1, history2 = main(a)
    save_csv(history2, 'valid', a.version)
    # save_csv(history1, 'train', a.version)