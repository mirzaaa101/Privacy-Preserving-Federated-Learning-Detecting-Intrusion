from functools import reduce
from typing import List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.common import Weights, FitRes, parameters_to_weights
from flwr.server.client_proxy import ClientProxy


class FedSum(FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None
        # Convert results
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]
        return aggregate(weights_results)


def aggregate(results: List[Tuple[Weights, int]]) -> Weights:
    """Compute weighted average."""
    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [[layer for layer in weights] for weights, _ in results]

    # Compute average weights of each layer
    weights_prime: Weights = [
        reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


if __name__ == "__main__":
    strategy = FedSum(
        min_fit_clients=2,
        min_available_clients=2,
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config={"num_rounds": 10, "round_timeout": None},
        strategy=strategy,
    )
