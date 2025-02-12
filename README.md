# Training

`train.py` is a script for training a regression model using PyTorch.
The script performs the following tasks:

1. Imports necessary modules and configurations, including PyTorch, data utilities, and model definitions.
2. Sets up the dataset using a custom MyRegressionDataset class, and creates a DataLoader for batching.
3. Initializes a RegressionModel, loss function (Mean Squared Error), optimizer (Adam), and learning rate scheduler (StepLR).
4. Iteratively trains the model over a specified number of epochs:
   - Sets the model in training mode.
   - Loops over batches of data, performing forward and backward passes.
   - Computes the loss and updates model parameters.
   - Saves the model state if it achieves a lower loss than previous epochs.
   - Adjusts the learning rate using the scheduler.
5. Optionally includes commented-out code for model evaluation on a validation set.
6. Outputs the loss for each training epoch.


# Converting to ONNX
