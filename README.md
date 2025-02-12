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
5. Outputs the loss for each training epoch.


# Converting to ONNX

`onnx_conv.py` is a script for converting a PyTorch regression model to ONNX format and running inference with both models.
The script performs the following tasks:

1. Sets up the dataset using a custom MyRegressionDataset class, and creates a DataLoader for batching.
2. Initializes a RegressionModel, loads the best model weights from training, and evaluates the model.
3. Converts the model to ONNX format using PyTorch's tracing functionality.
4. Creates an InferenceSession for both the original PyTorch model and the converted ONNX model, and loops over the data, running inference with both models and comparing the Mean Squared Error (MSE) and Mean Absolute Error (MAE) results.


# Quantization

`onnx_quant.py` is a script for quantizing an ONNX regression model using onnxruntime's quantization functionality.
The script performs the following tasks:

1. Sets up the dataset using a custom MyRegressionDataset class, and creates a DataLoader for batching.
2. Initializes a QuantizationDataReader for reading the data and quantizing the model.
3. Quantizes the model using onnxruntime's dynamic quantization.
