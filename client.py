# Federated Learning client for brain tumor segmentation using Flower and PyTorch
# This implementation includes top-k gradient sparsification to reduce communication cost.
import flwr as fl
import torch
from segmentation_Unet import DynamicUNet, BrainTumorClassifier
from dataset import double_data_load
from utils import get_model_weights, set_model_weights, get_topk_deltas
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNetFLClient(fl.client.NumPyClient):
    """
        Federated learning client implementation for 2D MRI brain tumor segmentation.
        Each client trains a local DynamicUNet model and communicates sparsified weight updates.
    """
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = DynamicUNet(filters=[32, 64, 128, 256, 512])
        self.classifier = BrainTumorClassifier(self.model, device=DEVICE)

        train1, val1, train2, val2, _ = double_data_load()
        if client_id == "0":
            self.trainloader = train1
            self.valloader = val1
        else:
            self.trainloader = train2
            self.valloader = val2

    def fit(self, parameters, config=None):
        """
            Performs one round of local training and returns sparsified model updates.
        """
        print(f"\n[Client {self.client_id}] Starting training...")

        # 1. Nastav a ulož váhy pred tréningom
        set_model_weights(self.model, parameters)
        self.base_weights = get_model_weights(self.model)

        # 2. Trénuj
        self.classifier.train(
            trainloader=self.trainloader,
            valloader=self.valloader,
            epochs=1,
            mini_batch_log=1
        )

        # 3. Sparsifikuj rozdiely
        deltas = get_topk_deltas(self.model, self.base_weights, k_ratio=0.01)

        return deltas, len(self.trainloader.dataset), {}  # ✅ bez "base_weights"

    def evaluate(self, parameters, config=None):
        """
            Evaluation is skipped in this minimal FL setup.
        """
        set_model_weights(self.model, parameters)
        print(f"\n[Client {self.client_id}] Skipping evaluation.")
        return 0.0, 100, {}

def main():
    client_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    fl.client.start_numpy_client(server_address="localhost:8080", client=UNetFLClient(client_id))

if __name__ == "__main__":
    main()
