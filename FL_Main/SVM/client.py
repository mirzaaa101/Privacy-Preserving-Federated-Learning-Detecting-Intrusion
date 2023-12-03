import random
import pickle
import pathlib
import argparse
import warnings
import typing as T

import flwr as fl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import log_loss

random.seed(42)
np.random.seed(42)


ROOT = pathlib.Path(__file__).resolve().parents[1]


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

        # Because one client has only two classses, we need to create a fake data for missing class
        for i in range(3):
            if i not in np.unique(self.y_train):
                self.X_train = np.vstack((self.X_train, self.X_train.mean(axis=0)))
                self.y_train = np.hstack((self.y_train, i))
            if i not in np.unique(self.y_test):
                self.X_test = np.vstack((self.X_test, self.X_test.mean(axis=0)))
                self.y_test = np.hstack((self.y_test, i))

        # Standardize the data
        scaler = np.load(ROOT / "data/scaler.npy", allow_pickle=True).item()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        # Model
        self.local_epochs = 10
        self.model = LinearSVC(
            max_iter=self.local_epochs,  # local epoch
        )
        self.model.classes_ = np.array([0, 1, 2])
        self.model.coef_ = np.zeros((3, 78))
        if self.model.fit_intercept:
            self.model.intercept_ = np.zeros((3,))

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
        return [self.enc_operator(layer) for layer in weights]

    def decrypt(self, weights: T.List[np.ndarray]) -> T.List[np.ndarray]:
        """Decrypt the weights using Paillier encryption.

        Args:
            weights (List[np.ndarray]): List of encrypted weights.

        Returns:
            List[np.ndarray]: List of decrypted weights.
        """
        if self.nohomo:
            return weights
        return [self.dec_operator(layer) for layer in weights]

    def get_parameters(self) -> T.List[np.ndarray]:
        """Get the model parameters.

        Returns:
            List[np.ndarray]: List of encrypted weights.
        """
        params = [np.array([len(self.X_train)]), self.model.coef_ * len(self.X_train)]
        if self.model.fit_intercept:
            params.append(self.model.intercept_ * len(self.X_train))
        return self.encrypt(params)

    def set_parameters(self, parameters: T.List[np.ndarray]) -> None:
        """Set the model parameters.

        Args:
            parameters (List[np.ndarray]): List of encrypted weights.
        """
        parameters = self.decrypt(parameters)
        num_examples_total = parameters[0][0]
        self.model.coef_ = parameters[1] / num_examples_total
        if self.model.fit_intercept:
            self.model.intercept_ = parameters[2] / num_examples_total

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
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

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
        accuracy = self.model.score(self.X_test, self.y_test)
        self.result["accuracy"].append(accuracy)

        # Save the result
        with open(f"result-{self.idx}-{self.nohomo}.pkl", "wb") as f:
            pickle.dump(self.result, f)

        # Save the model state dict
        with open(f"model-{self.idx}-{self.nohomo}.pkl", "wb") as f:
            pickle.dump(self.model, f)
        return 1.0, len(self.X_test), {"accuracy": accuracy}


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--client_idx", type=int, required=True, help="Client index")
parser.add_argument(
    "--nohomo", action="store_true", help="Disable homomorphic encryption"
)

if __name__ == "__main__":
    args = parser.parse_args()
    fl.client.start_numpy_client(
        server_address="localhost:8081", client=Client(args.client_idx, args.nohomo)
    )
