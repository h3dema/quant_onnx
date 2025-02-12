import torch
import torch.nn as nn

from config import n_features, n_outputs

class RegressionModel(nn.Module):
    """ Simple neural network model for regression tasks
    """
    def __init__(self, n_features, n_outputs):
        """
        Initializes the RegressionModel with a sequential neural network architecture.

        Args:
            n_features (int): The number of input features.
            n_outputs (int): The number of output features.
        """
        super(RegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_outputs)
        )

    def forward(self, x):
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.
        """
        return self.model(x)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset import MyRegressionDataset
    from config import batch_size
    dataset = MyRegressionDataset(n_features, n_outputs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RegressionModel(n_features, n_outputs)

    data = next(iter(dataloader))
    # x [batch_size, n_features]
    # y [batch_size, n_outputs]
    x, y = data
    assert y.shape[1] == n_outputs, f"y shape is not correct: {y.shape}"

    yhat = model(x)

    assert yhat.shape[1] == n_outputs, f"\u0177 shape is not correct: {yhat.shape}"
    print("\u0177:", yhat.shape)
