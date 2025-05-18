# server.py
# Flower server with a custom federated averaging strategy (FedAvgTopK)
import torch
from flwr.server.strategy import FedAvg
import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from typing import List, Dict, Optional, Tuple
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters
from segmentation_Unet import DynamicUNet
from utils import set_model_weights
from dataset import double_data_load
from test import evaluate_model_on_loader


class FedAvgTopK(FedAvg):
    """
        Custom FedAvg strategy that allows tracking and updating current global weights.
        In this version, all model updates (full weights) are aggregated via weighted average.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_weights = None # Keeps global model weights between rounds

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        print(f"[Server] Aggregating round {server_round} with {len(results)} clients...")

        if failures:
            print(f"Warning: {len(failures)} clients failed")

        if not results:
            print("No results to aggregate")
            return None, {}

        # Convert parameter format to NumPy arrays
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Initialize global weights if not yet set
        if self.current_weights is None:
            print("Initializing global weights from first client")
            self.current_weights = weights_results[0][0]  # Use first client's weights as base

        # Aggregate weights using weighted average (FedAvg logic)
        total_examples = sum([num_examples for _, num_examples in weights_results])
        print(f"Total training examples this round: {total_examples}")

        # Aggregate deltas (client sends full weights in this implementation)
        agg_weights = [
            np.zeros_like(layer) for layer in self.current_weights
        ]
        for weights, num_examples in weights_results:
            for i, layer in enumerate(weights):
                agg_weights[i] += layer * (num_examples / total_examples)

        self.current_weights = agg_weights

        print("Aggregation complete")
        return ndarrays_to_parameters(self.current_weights), {}

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        return None  # Skip evaluation


def main():
    strategy = FedAvgTopK(
        fraction_fit=1.0,  # Use all available clients
        min_fit_clients=2,  # Minimum 2 clients needed
        min_available_clients=2,  # Wait until 2 clients are available
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy
    )

    # evaluation after training
    global_weights = strategy.current_weights
    model = DynamicUNet(filters=[32, 64, 128, 256, 512])
    set_model_weights(model, global_weights)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    _, _, _, _, test_loader = double_data_load()
    evaluate_model_on_loader(model, test_loader, device=device)

    # save model
    state_dict = model.state_dict()
    new_state_dict = {
        k: torch.tensor(w, dtype=state_dict[k].dtype) for k, w in zip(state_dict.keys(), global_weights)
    }
    model.load_state_dict(new_state_dict)

    torch.save(model.state_dict(), "best_model.pth")


if __name__ == "__main__":
    main()