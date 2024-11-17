import gc
from collections import OrderedDict
from pathlib import Path

import medmnist
import numpy as np
import toml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

fds = {}

configs = toml.load("pyproject.toml")


class CustomResNet(nn.Module):
    def __init__(self, in_channels=0, num_classes=0):
        super(CustomResNet, self).__init__()
        self.num_classes = num_classes

        # Define convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.GroupNorm(32, 64)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, 128)

        # Define residual blocks
        self.res1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 128),
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 256),
        )

        self.res3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 512),
        )

        # Define final classification layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Initial convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Residual blocks
        residual = x
        x = self.res1(x)
        x += residual
        x = self.res2(x)
        x = self.res3(x)

        # Final classification layer
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def create_splits(number: int, split: int = 11, ratio: float = 0.75, seed: int = 42):
    """Function used to create splits for federated datasets.

    Args:
        number: The number to split.
        split: Number of splits to do.
        ratio: Determine the minimum and maximum size of one split.
        seed: Random seed.

    Returns:
        splits: The list of elements per split.
        added_splits: Number of elements per splits matched to index.
    """

    # Set the seed to always get the same splits for evaluation purposes
    np.random.seed(seed)
    # Contains number of elements per split
    splits = []
    # Contains cumulated sum of splits to match indexes
    added_splits = []
    entire_part = number // split
    # A single split cannot be lower than entire_part - min_split
    min_split = entire_part * ratio
    if number < split:
        return [number]
    for i in range(split):
        if number % split != 0 and i >= split - (number % split):
            splits.append(entire_part + 1)
        else:
            splits.append(entire_part)
    length = len(splits) if len(splits) % 2 == 0 else len(splits) - 1
    for s in range(0, length, 2):
        random_value = np.random.randint(low=0, high=min_split)
        splits[s] -= random_value
        added_splits.append(int(np.sum(splits[:s])))
        splits[s + 1] += random_value
        added_splits.append(int(np.sum(splits[: s + 1])))
    if len(splits) % 2 != 0:
        added_splits.append(np.sum(splits[:-1]))
    added_splits.append(np.sum(splits))
    return splits, added_splits


def create_datasets(dataset_name, num_partitions, batch_size):
    """Split the whole dataset in IID or non-IID manner for distributing to clients,
    merging labels with data and returning DataLoaders for batch processing.
    """
    if dataset_name not in medmnist.INFO.keys():
        # dataset not found exception
        error_message = f'Dataset "{dataset_name}" is not supported or cannot be found in TorchVision Datasets!'
        raise AttributeError(error_message)

    transform = Compose(
        [
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    info = medmnist.INFO[dataset_name]
    task = info["task"]
    if task == "multi-label, binary-class":
        raise "datasets with task='multi-label, binary-class' are not supported"

    DataClass = getattr(medmnist, info["python_class"])

    # Define the folder name

    # Create the folder if it doesn't exist
    Path("./data").mkdir(parents=True, exist_ok=True)
    # Load training and test datasets

    training_dataset = DataClass(
        root="./data", split="train", transform=transform, download=True
    )
    test_dataset = DataClass(
        root="./data", split="test", transform=transform, download=True
    )

    # Split the training dataset for the clients
    _, added_splits = create_splits(
        len(training_dataset), split=num_partitions, ratio=0.50
    )

    local_datasets = []
    for c in range(num_partitions):
        client_dataset = Subset(
            training_dataset,
            range(int(added_splits[c]), int(added_splits[c + 1])),
        )

        # Split into 80% training and 20% validation
        train_len = int(0.8 * len(client_dataset))
        val_len = len(client_dataset) - train_len
        client_train_dataset, client_val_dataset = random_split(
            client_dataset, [train_len, val_len]
        )

        # Create DataLoaders for both training and validation datasets
        client_train_loader = DataLoader(
            client_train_dataset, batch_size=batch_size, shuffle=True
        )
        client_val_loader = DataLoader(
            client_val_dataset, batch_size=batch_size, shuffle=False
        )

        local_datasets.append((client_train_loader, client_val_loader))

    # Create DataLoader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    labels = [*info["label"].values()]
    return task, local_datasets, test_loader, labels


def load_data(partition_id: int, num_partitions: int, dataset_name: str):
    """
    Load data for a specific partition. If the global variable `fds` is empty,
    it creates and partitions the dataset using `create_datasets` and then returns
    the specific partition dataset.

    Args:
    - partition_id: The ID of the partition to load.
    - num_partitions: Total number of partitions (clients) in the dataset.
    - dataset_name: The name of the dataset to load and partition.
    - num_clients: Total number of clients to split the dataset.

    Returns:
    - A tuple with (task, train_dataset, val_dataset, test_dataset, labels).
    """
    global fds

    # If fds is empty, create datasets and partition them
    if not fds:
        task, local_datasets, test_dataset, labels = create_datasets(
            dataset_name, num_partitions, 32
        )
        fds["task"] = task
        fds["local_datasets"] = local_datasets
        fds["global_test_dataset"] = test_dataset
        fds["labels"] = labels
        fds["num_partitions"] = num_partitions

    # Ensure the partition_id is valid
    if partition_id >= num_partitions or partition_id < 0:
        raise ValueError(
            f"Partition ID {partition_id} is out of range for the total partitions {num_partitions}."
        )

    # Get the training and validation datasets for the specified partition
    train_dataset, val_dataset = fds["local_datasets"][partition_id]

    return (
        fds["task"],
        train_dataset,
        val_dataset,
        fds["labels"],
        medmnist.INFO[dataset_name]["n_channels"],
    )


def get_centralized_eval_dataset(dataset_name):
    if dataset_name not in medmnist.INFO.keys():
        # dataset not found exception
        error_message = f'Dataset "{dataset_name}" is not supported or cannot be found in TorchVision Datasets!'
        raise AttributeError(error_message)

    transform = Compose(
        [
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    info = medmnist.INFO[dataset_name]
    task = info["task"]
    DataClass = getattr(medmnist, info["python_class"])

    # Define the folder name

    # Create the folder if it doesn't exist
    Path("./data").mkdir(parents=True, exist_ok=True)

    test_dataset = DataClass(
        root="./data", split="test", transform=transform, download=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    return task, test_dataloader


def train(net, train_loader, val_loader, epochs, device, task):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for _ in range(epochs):
        for batch in train_loader:
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            labels = labels.squeeze().long()
            loss = criterion(net(data), labels)
            loss.backward()
            optimizer.step()
    net.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    train_loss, train_acc = test(net, train_loader, device, task)
    val_loss, val_acc = test(net, val_loader, device, task)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


def test(net, test_loader, device, task, desc="Testing"):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc):
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            outputs = net(data)
            labels = labels.squeeze().long()
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    net.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    accuracy = correct / len(test_loader.dataset)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def get_configs():
    return configs


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
