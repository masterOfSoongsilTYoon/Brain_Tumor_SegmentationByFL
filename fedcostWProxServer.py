from typing import Callable, Dict, List, Optional, Tuple, Union, OrderedDict
import flwr as fl
from flwr.common import FitRes, MetricsAggregationFn, NDArrays, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from networks import Baseline_net
from client import set_parameters
from train import eval, parserer
from dataset import CustomDataset
import torch
import pandas as pd
from utils import save_csv
import warnings
import numpy as np
import os
import random

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from functools import reduce
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
a= parserer()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ori_aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def global_loss_PIDAVG(lis, idx):
    return sum([float((lis[-2][id][0][0]))/float((lis[-1][id][0][0])) for id in range(idx)])

def client_loss_PIDAVG(lis, idx):
    return [float((lis[-2][id][0][0]))/float((lis[-1][id][0][0])) for id in range(idx)]

def global_iou_PIDAVG(lis, idx):
    return sum([float((lis[-1][id][1]["dice"][0]))/float((lis[-2][id][1]["dice"][0])) for id in range(idx)])

def client_iou_PIDAVG(lis, idx):
    return [float((lis[-1][id][1]["dice"][0]))/float((lis[-2][id][1]["dice"][0])) for id in range(idx)]

def evaluate_fn(roundn, parameters, config):
    net = Baseline_net(1, a.mode, a.data).to(DEVICE)
    set_parameters(net, parameters)
    test_df = pd.read_csv(f'./CSV/brain_valid_central.csv')
    test_data = CustomDataset(test_df, transform=False, mode="brain")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0, collate_fn = lambda x: x)
    history=eval(net, test_loader, torch.nn.BCEWithLogitsLoss().to(DEVICE), False, data="brain", a=a)
    history["mIOU_acc"] = {"mIOU": history["mIOU"], "acc": history["acc"], "f1Score":history['f1Score'], "dice":history["dice"]}
    return history['loss'], history["mIOU_acc"]
    
def aggregate(results: List[Tuple[NDArrays, int]], client_lis, round) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_ = [num_examples for _, num_examples in results]
    num_examples_total = sum(num_examples_)
    client_num = len(num_examples_)
    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]
    if round>2:
        weighted_loss_layer=[
         [layer*loss for layer in weights] for (weights, _), loss in zip(results, client_loss_PIDAVG(client_lis, client_num))
        ]
        weighted_dice_layer=[
         [layer*metric for layer in weights] for (weights, _), metric in zip(results, client_iou_PIDAVG(client_lis, client_num))
        ]
        
    # Compute average weights of each layer
        weights_prime: NDArrays = [
            (0.5*reduce(np.add, layer_updates) / num_examples_total) 
            +(0.5*reduce(np.add, loss_layer)/global_loss_PIDAVG(client_lis, client_num))
            # +(0.25*reduce(np.add, dice_layer)/global_iou_PIDAVG(client_lis, client_num))
            for layer_updates, loss_layer, _ in zip(zip(*weighted_weights), zip(*weighted_loss_layer), zip(*weighted_dice_layer))
        ]
        
    else:
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total 
            for layer_updates in zip(*weighted_weights)
        ]
    return weights_prime

class Custom_straegy(fl.server.strategy.FedAvg):
    def __init__(self, *,net, fraction_fit: float = 1, fraction_evaluate: float = 1, min_fit_clients: int = 2, min_evaluate_clients: int = 2, min_available_clients: int = 2, evaluate_fn: Callable[[int, NDArrays, Dict[str, Scalar]], Tuple[float, Dict[str, Scalar]] | None] | None = None, on_fit_config_fn: Callable[[int], Dict[str, Scalar]] | None = None, on_evaluate_config_fn: Callable[[int], Dict[str, Scalar]] | None = None, accept_failures: bool = True, initial_parameters: Parameters | None = None, fit_metrics_aggregation_fn: MetricsAggregationFn | None = None, evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None) -> None:
        self.net = net
        self.global_lis = []
        self.client_lis = []
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Tuple[ClientProxy, FitRes] | BaseException]) -> Tuple[Parameters | None, Dict[str, Scalar]]:
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        if server_round>0:
            self.client_lis.append([
            evaluate_fn(server_round, weights,{}) for weights, num_examples in weights_results
            ])
        # parameter = ndarrays_to_parameters(ori_aggregate(weights_results))
        # parameter = parameters_to_ndarrays(parameter)
        # loss,metric=evaluate_fn(server_round, parameter, {})
        # self.global_lis.append((loss,metric))
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results, self.client_lis, server_round))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        aggregated_parameters,aggregated_metrics= parameters_aggregated, metrics_aggregated
        
        
        
        
        # aggregated_parameters, aggregated_metrics=super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            set_parameters(self.net, aggregated_ndarrays)

            # Save the model
            # torch.save(self.net.state_dict(), f"./models/{a.version}/model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics
    def evaluate(self, server_round: int, parameters: Parameters) -> Tuple[float, Dict[str, Scalar]] | None:
        return super().evaluate(server_round, parameters)

if __name__ == "__main__":
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    torch.cuda.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    
    if os.path.exists('models') is False:
        os.makedirs('models')

    save_path = 'models/' + a.version
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    print(DEVICE, " 사용중")
    
    warnings.filterwarnings(action='ignore')
    history=fl.server.start_server(
        server_address="203.253.25.173:8085",
        config= fl.server.ServerConfig(num_rounds=200),
        strategy=Custom_straegy(net=Baseline_net(1,data="brain"),min_fit_clients=5, min_available_clients=5, min_evaluate_clients=5, evaluate_fn=evaluate_fn),
        
    )
    
    # print("loss centralized: ",history.losses_centralized)
    # print("loss distributed: ",history.losses_distributed)
    # print("acc centralized: ",history.metrics_centralized)
    # print("acc distributed: ",history.metrics_distributed)
    # print("acc distributed fit: ",history.metrics_distributed_fit)
    
    csv_history={
        "loss" :[z[1][0] for z in history.losses_centralized],
        "acc" : [z[1][0] for z in history.metrics_centralized["acc"]],
        "mIOU": [z[1][0] for z in history.metrics_centralized["mIOU"]],
        "f1Score": [z[1][0] for z in history.metrics_centralized["f1Score"]],
        "dice": [z[1][0] for z in history.metrics_centralized["dice"]]
    }
    save_csv(csv_history, mode="train", name="Brain_FL_Prox_costW")
    