import gc
from typing import Dict, List, Optional, Tuple

import torch
from flwr.common import Context, Metrics, NDArrays, Scalar, ndarrays_to_parameters
from flwr.server import (
    Driver,
    LegacyContext,
    ServerApp,
    ServerConfig,
)
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from src.task import (
    CustomResNet,
    get_centralized_eval_dataset,
    get_configs,
    get_weights,
    set_weights,
    test,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_config = get_configs()["dataset"]


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    examples = [num_examples for num_examples, _ in metrics]
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_accuracies = [
        num_examples * m["train_accuracy"] for num_examples, m in metrics
    ]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]

    return {
        "train_loss": sum(train_losses) / sum(examples),
        "train_accuracy": sum(train_accuracies) / sum(examples),
        "val_loss": sum(val_losses) / sum(examples),
        "val_accuracy": sum(val_accuracies) / sum(examples),
    }


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    task, test_loader = get_centralized_eval_dataset(dataset_config["name"])

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        set_weights(model, parameters)  # Update model with the latest parameters
        loss, accuracy = test(
            model, test_loader=test_loader, device=DEVICE, task=task, desc="Evaluating"
        )
        gc.collect()
        return loss, {"accuracy": accuracy}

    return evaluate


app = ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:

    # Initialize global model
    net = CustomResNet(dataset_config["in_channels"], dataset_config["num_classes"])
    model_weights = get_weights(net)
    parameters = ndarrays_to_parameters(model_weights)

    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    # Note: The fraction_fit value is configured based on the DP hyperparameter `num-sampled-clients`.
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
        evaluate_fn=get_evaluate_fn(net),
    )

    # Construct the LegacyContext
    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # Create the train/evaluate workflow
    workflow = DefaultWorkflow(
        fit_workflow=SecAggPlusWorkflow(
            num_shares=context.run_config["num-shares"],
            reconstruction_threshold=context.run_config["reconstruction-threshold"],
            max_weight=context.run_config["max-weight"],
        )
    )

    # Execute
    workflow(driver, context)


# def server_fn(context: Context) -> ServerAppComponents:
#     net = CustomResNet(dataset_config["in_channels"], dataset_config["num_classes"])
#     ndarrays = get_weights(net)
#     parameters = ndarrays_to_parameters(ndarrays)
#
#     num_rounds = context.run_config["num-server-rounds"]
#     fraction_fit = context.run_config["fraction-fit"]
#     fraction_evaluate = context.run_config["fraction-evaluate"]
#     strategy = FedAvg(
#         fraction_fit=fraction_fit,
#         fraction_evaluate=fraction_evaluate,
#         fit_metrics_aggregation_fn=weighted_average,
#         initial_parameters=parameters,
#         evaluate_fn=get_evaluate_fn(net),
#     )
#     config = ServerConfig(num_rounds=num_rounds)
#
#     return ServerAppComponents(config=config, strategy=strategy)
#
# app = ServerApp(server_fn=server_fn)
