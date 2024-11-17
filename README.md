# Differentially Private Federated Learning with Secure Aggregation using Flower on MedMNIST

This project demonstrates an experiment in **Federated Learning** using the **Flower** framework, incorporating sample-level **Differential Privacy** and enabling **Secure Aggregation** through the **SecAgg+** protocol. The experiment is conducted on the **MedMNIST** dataset collection.

## Setup

This project is built and tested on **Python 3.8.10**.

In the project's main directory, run the following commands to create a virtual environment and install the required packages:

```bash
python -m venv env
```

```bash
source env/bin/activate
```

```bash
python -m pip install .
```

## Key Features

1. **Local Differential Privacy (LocalDP)**:
   - Differential privacy is implemented using Flower's [LocalDpMod](https://flower.ai/docs/framework/ref-api/flwr.client.mod.LocalDpMod.html).

2. **Secure Aggregation (SecAgg+ Protocol)**:
   - The SecAgg+ protocol is implemented via Flower's [secaggplus_mod](https://flower.ai/docs/framework/ref-api/flwr.client.mod.secaggplus_mod.html).

3. **Easy Parameter Control**:
   - Parameters related to federated learning settings and the SecAgg+ protocol can be controlled from the `pyproject.toml` file.
