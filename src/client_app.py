import gc

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import LocalDpMod, parameters_size_mod, secaggplus_mod
from flwr.common import Context

from src.task import (
    CustomResNet,
    get_configs,
    get_weights,
    load_data,
    set_weights,
    test,
    train,
)


class FlowerClient(NumPyClient):
    def __init__(
        self,
        client_id,
        train_loader,
        test_loader,
        in_channels,
        num_classes,
        task,
        local_epochs,
    ) -> None:
        # self.net = Net()
        self.client_id = client_id
        self.net = CustomResNet(in_channels=in_channels, num_classes=num_classes)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.task = task
        self.local_epochs = local_epochs

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.train_loader,
            self.test_loader,
            epochs=self.local_epochs,
            device=self.device,
            task=self.task,
        )
        weights = get_weights(self.net)
        gc.collect()
        return weights, len(self.train_loader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.test_loader, self.device, task=self.task)
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    task, train_loader, test_loader, labels, in_channels = load_data(
        partition_id,
        context.node_config["num-partitions"],
        get_configs()["dataset"]["name"],
    )
    return FlowerClient(
        partition_id,
        train_loader,
        test_loader,
        in_channels=in_channels,
        num_classes=len(labels),
        task=task,
        local_epochs=context.run_config["local-epochs"],
    ).to_client()


local_dp_mod = LocalDpMod(
    clipping_norm=7.0, sensitivity=1.0, epsilon=6.0, delta=(1.9 * 10 ** -4)
)
# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[local_dp_mod, secaggplus_mod, parameters_size_mod],
)
