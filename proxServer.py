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
a= parserer()
def evaluate_fn(roundn, parameters, config):
    net = Baseline_net(3, a.mode, a.data).to(DEVICE)
    set_parameters(net, parameters)
    test_df = pd.read_csv(f'./CSV/brain_valid_central.csv')
    test_data = CustomDataset(test_df, transform=False, mode="brain")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0, collate_fn = lambda x: x)
    history=eval(net, test_loader, torch.nn.BCEWithLogitsLoss().to(DEVICE), False, data="brain", a=a)
    history["mIOU_acc"] = {"mIOU": history["mIOU"], "acc": history["acc"], "f1Score":history['f1Score'], "dice":history["dice"]}
    return history['loss'], history["mIOU_acc"]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Custom_straegy(fl.server.strategy.FedProx):
    def __init__(self, *,net, fraction_fit: float = 1, fraction_evaluate: float = 1, min_fit_clients: int = 2, min_evaluate_clients: int = 2, min_available_clients: int = 2, evaluate_fn: Callable[[int, NDArrays, Dict[str, Scalar]], Tuple[float, Dict[str, Scalar]] | None] | None = None, on_fit_config_fn: Callable[[int], Dict[str, Scalar]] | None = None, on_evaluate_config_fn: Callable[[int], Dict[str, Scalar]] | None = None, accept_failures: bool = True, initial_parameters: Parameters | None = None, fit_metrics_aggregation_fn: MetricsAggregationFn | None = None, evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None, proximal_mu: float) -> None:
        self.net = net
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, proximal_mu=proximal_mu)
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Tuple[ClientProxy, FitRes] | BaseException]) -> Tuple[Parameters | None, Dict[str, Scalar]]:
        aggregated_parameters, aggregated_metrics=super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # # Convert `Parameters` to `List[np.ndarray]`
            # aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # # Convert `List[np.ndarray]` to PyTorch`state_dict`
            # set_parameters(self.net, aggregated_ndarrays)

            # # Save the model
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
        server_address="203.253.25.173:8082",
        config= fl.server.ServerConfig(num_rounds=200),
        strategy=Custom_straegy(net=Baseline_net(3,data="brain"),min_fit_clients=5, min_available_clients=5, min_evaluate_clients=5, evaluate_fn=evaluate_fn, proximal_mu=0.3),
        
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
    save_csv(csv_history, mode="train", name="Brain_FL_Prox_NonePartial")
    
    # ploting(history.metrics_centralized, mode="acc", roundn=len(history.metrics_centralized["acc"]))
    # ploting({"loss":history.losses_centralized}, mode="loss", roundn=len(history.losses_centralized))