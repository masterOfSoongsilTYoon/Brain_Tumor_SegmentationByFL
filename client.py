import os

from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
from networks import Baseline_net
from dataset import CustomDataset
from train import train, eval
import flwr as fl
import torch 
from collections import OrderedDict
import argparse
import pandas as pd
import numpy as np
from train import parserer
import random
import warnings
from loss import FocalLoss
# parser = argparse.ArgumentParser()
# parser.add_argument('--epochs', default=1, type=int, metavar='N',
#                     help='number of total epochs to run')

# parser.add_argument('--num', default=1, type=int, metavar='N',
#                     help='number of client id(1~10)')

# parser.add_argument('--batch', default=32, type=int, metavar='N',
#                     help='number of client id(1~10)')

# a = parser.parse_args()

def set_parameters(net:torch.nn.Module, parameters:list, *kargs) -> None:
    # Set model parameters from a list of NumPy ndarrays
    keys = [k for k in net.state_dict().keys()]
    params_dict = zip(keys, parameters)
    state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=False)
        

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, validloader, DEVICE, client_num:int, a=None) -> None:
        super(FlowerClient, self).__init__()
        self.net = net
        self.train_loader = trainloader
        self.valid_loader = validloader
        # self.test_loader = testloader
        self.DEVICE= DEVICE
        self.id = client_num
        self.a = a
    def get_parameters(self, config, *kargs):
        # Return model parameters as a list of NumPy ndarrays, excluding parameters of BN layers when using FedBN
        return [val.cpu().numpy() for name, val in self.net.state_dict().items()]
    # if 'bn' not in k
    def set_parameters(self, parameters:list, *kargs) -> None:
        # Set model parameters from a list of NumPy ndarrays
        keys = [k for k in self.net.state_dict().keys()]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=False)
        
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.train_loader, torch.nn.BCELoss().to(DEVICE), torch.optim.SGD(self.net.parameters(), self.a.lr), None, self.a, False, data=a.data)
        return self.get_parameters({}), len(self.train_loader.dataset),{}
    def evaluate(self, parameters, *kargs) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        history = eval(self.net, self.valid_loader,torch.nn.BCEWithLogitsLoss().to(DEVICE), False, data="brain",a=self.a)
        return history["loss"][0], len(self.valid_loader.dataset), {"accuracy": history["mIOU"][0]}
    
if __name__ == "__main__":
    a = parserer()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    torch.cuda.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    warnings.filterwarnings(action='ignore')
    
    net = Baseline_net(1, "unet", "brain").to(DEVICE)
    if a.transfered is not None:
        net.load_state_dict(torch.load(a.transfered))
    client_id = a.num
    df = pd.read_csv(f'./CSV/brain_train_client{client_id}.csv')
    valid_df = pd.read_csv(f'./CSV/brain_valid_client{client_id}.csv')
    train_dataset = CustomDataset(df, transform=False, mode="brain")
    valid_dataset = CustomDataset(valid_df, transform=False, mode="brain")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=a.batch_size, shuffle=True, num_workers=0, collate_fn = lambda x: x)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=a.batch_size, shuffle=False, num_workers=0, collate_fn = lambda x: x)
    fl.client.start_numpy_client(server_address="[:,:]:8085", client=FlowerClient(net, train_loader, valid_loader, DEVICE, client_id, a=a))