import random
import pickle
import pathlib
import argparse
import typing as T

import torch
import flwr as fl
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

ROOT = pathlib.Path(__file__).resolve().parents[1]


class MyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = torch.nn.Linear(78, 12)
        self.fc2 = torch.nn.Linear(12, 6)
        self.fc3 = torch.nn.Linear(6, 3)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.softmax(self.fc3(x), dim=1)


class Client(fl.client.NumPyClient):
    """Simple client implementing `NumPyClient`.

    Args:
        idx (int): Index of the client.
        nohomo (bool): Disable homomorphic encryption.
    """

    def __init__(self, idx: int, nohomo: bool) -> None:
        super().__init__()

        self.idx = idx
        self.nohomo = nohomo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # Load the dataset
        label2id = {
            "DDoS": 0,
            "PortScan": 1,
            "BENIGN": 2,
        }
        df = pd.read_csv(ROOT / f"data/dataset_{self.idx}.csv")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        df["Label"] = df["Label"].map(label2id)
        Y = df["Label"].values
        X = df.drop(columns=["Label", "IP"]).values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        # Standardize the data
        scaler = np.load(ROOT / "data/scaler.npy", allow_pickle=True).item()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        self.X_train = torch.tensor(self.X_train).float().to(self.device)
        self.X_test = torch.tensor(self.X_test).float().to(self.device)
        self.y_train = torch.tensor(self.y_train).long().to(self.device)
        self.y_test = torch.tensor(self.y_test).long().to(self.device)

        # Model
        self.local_epochs = 1
        self.model = MyModel().to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.result = {
            "accuracy": [],
            "loss": [],
        }

        # Homomorphic encryption
        public_key = np.load(ROOT / "data/public_key.npy", allow_pickle=True).item()
        private_key = np.load(ROOT / "data/private_key.npy", allow_pickle=True).item()
        self.enc_operator = np.frompyfunc(public_key.encrypt, 1, 1)
        self.dec_operator = np.frompyfunc(private_key.decrypt, 1, 1)

    def encrypt(self, weights: T.List[np.ndarray]) -> T.List[np.ndarray]:
        """Encrypt the weights using Paillier encryption.

        Args:
            weights (List[np.ndarray]): List of weights.

        Returns:
            List[np.ndarray]: List of encrypted weights.
        """
        if self.nohomo:
            return weights
        return weights[:-1] + [self.enc_operator(weights[-1])]
        # return [self.enc_operator(layer) for layer in weights]

    def decrypt(self, weights: T.List[np.ndarray]) -> T.List[np.ndarray]:
        """Decrypt the weights using Paillier encryption.

        Args:
            weights (List[np.ndarray]): List of encrypted weights.

        Returns:
            List[np.ndarray]: List of decrypted weights.
        """
        if self.nohomo:
            return weights
        return weights[:-1] + [self.dec_operator(weights[-1])]
        # return [self.dec_operator(layer) for layer in weights]

    def get_parameters(self) -> T.List[np.ndarray]:
        """Get the model parameters.

        Returns:
            List[np.ndarray]: List of encrypted weights.
        """
        self.model.eval()
        params = [
            v.cpu().float().numpy() * len(self.X_train)
            for _, v in self.model.state_dict().items()
        ]
        weights = [np.array([len(self.X_train)])] + params
        return self.encrypt(weights)

    def set_parameters(self, parameters: T.List[np.ndarray]) -> None:
        """Set the model parameters.

        Args:
            parameters (List[np.ndarray]): List of encrypted weights.
        """
        parameters = self.decrypt(parameters)
        num_examples_total = parameters[0][0]
        params_dict = zip(self.model.state_dict().keys(), parameters[1:])
        state_dict = OrderedDict(
            {k: torch.tensor(v.astype(np.float32)) / num_examples_total for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: T.List[np.ndarray], config: T.Dict[str, T.Any]
    ) -> T.Tuple[T.List[np.ndarray], int, T.Dict[str, T.Any]]:
        """Train the model on the locally held dataset.

        Args:
            parameters (List[np.ndarray]): List of encrypted weights.
            config (Dict[str, Any]): Configuration dictionary.

        Returns:
            Tuple[List[np.ndarray]: Tuple of updated model parameters.
            int: Number of examples used for training.
            Dict[str, Any]]: Dictionary with additional information (empty in this case).
        """
        self.set_parameters(parameters)
        self.model.train()

        for epoch in range(self.local_epochs):
            self.optimizer.zero_grad()
            outputs = self.model(self.X_train)

            loss = self.criterion(outputs, self.y_train)
            loss.backward()

            self.optimizer.step()
            self.result["loss"].append(loss.item())
        return self.get_parameters(), len(self.X_train), {}

    def evaluate(
        self, parameters: T.List[np.ndarray], config: T.Dict[str, T.Any]
    ) -> T.Tuple[int, int, T.Dict[str, T.Any]]:
        """Evaluate the locally held dataset against the provided parameters.

        Args:
            parameters (List[np.ndarray]): List of encrypted weights.
            config (Dict[str, Any]): Configuration dictionary.

        Returns:
            Tuple[int: Loss on the locally held dataset.
            int: Number of examples used for evaluation.
            Dict[str, Any]]: Dictionary with additional information (in this case accuracy).
        """
        self.set_parameters(parameters)
        self.model.eval()

        outputs = self.model(self.X_test)
        loss = self.criterion(outputs, self.y_test)

        predicted = outputs.argmax(1)
        accuracy = (predicted == self.y_test).sum().item() / len(self.y_test)
        self.result["accuracy"].append(accuracy)

        # Save the result
        with open(f"result-{self.idx}-{self.nohomo}.pkl", "wb") as f:
            pickle.dump(self.result, f)

        # Save the model state dict
        torch.save(self.model.state_dict(), f"model-{self.idx}-{self.nohomo}.pth")
        return loss.item(), len(self.X_test), {"accuracy": accuracy}


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--client_idx", type=int, required=True, help="Client index")
parser.add_argument(
    "--nohomo", action="store_true", help="Disable homomorphic encryption"
)

if __name__ == "__main__":
    args = parser.parse_args()
    fl.client.start_numpy_client(
        server_address="localhost:8080", client=Client(args.client_idx, args.nohomo)
    )
