"""

**Note**: if you notice a warning saying
UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names.Available providers: 'AzureExecutionProvider, CPUExecutionProvider'

you have to install `onnxruntime_gpu` using pip
"""
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

import onnx
import onnxruntime as ort

from config import n_features, n_outputs
from config import batch_size, model_out, best_model_name, onnx_model_name
from dataset import MyRegressionDataset
from model import RegressionModel
from onnx_utils import to_numpy


if __name__ == "__main__":

    # create dataset
    dataset = MyRegressionDataset(n_features, n_outputs)
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = RegressionModel(n_features, n_outputs)
    # Load the best model's weights
    model.load_state_dict(torch.load(os.path.join(model_out, best_model_name)))
    model.eval()

    # Trace the Model (Optional, Recommended)
    # Perform inference once using an input (can be dummy).
    data = next(iter(dataloader))
    x, y = data
    yhat = model(x)

    # Convert to ONNX
    onnx_model_fp32_path = os.path.join(model_out, onnx_model_name)

    # **Note**: Constant-folding will replace some of the ops that have all constant inputs with pre-computed constant nodes.
    torch.onnx.export(
        model,  # pytorch model
        x,  # model input
        onnx_model_fp32_path,  # path
        export_params=True,   # store the trained parameter weights inside the model file
        opset_version=14,  # the ONNX version to export the model to
        do_constant_folding=True,  # constant folding for optimization
        input_names = ['input'],   # input names
        output_names = ['output'],   # output names
        dynamic_axes={'input' : {0 : 'batch_size'},  # variable length axes
                      'output' : {0 : 'batch_size'}
                      },
    )

    # Verify the ONNX model’s structure and
    # confirm that the model has a valid schema.
    # The validity of the ONNX graph is verified by checking the model’s version,
    # the graph’s structure, as well as the nodes and their inputs and outputs.

    model_onnx = onnx.load(onnx_model_fp32_path)
    onnx.checker.check_model(model_onnx)  # An exception is raised if the test fails.

    # Run Inference
    print("Checking Pytorch vs. ONNX results")

    # Let's compare both models: Pytorch vs. ONNX
    ort_provider = ['CPUExecutionProvider']
    use_cuda = False  #  torch.cuda.is_available()
    if use_cuda:
        model.to('cuda')
        ort_provider = ['CUDAExecutionProvider']

    # Prepare the Models
    ort_sess = ort.InferenceSession(onnx_model_fp32_path, providers=ort_provider)

    total_ort_mse = 0
    total_ort_mae = 0
    total_pt_mse = 0
    total_pt_mae = 0
    for x, y in tqdm(dataloader, ascii=True, unit="batches"):

        ort_inputs = {ort_sess.get_inputs()[0].name: to_numpy(x)}
        ort_outs = ort_sess.run(None, ort_inputs)[0]

        # Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE)
        syy = (to_numpy(y) - ort_outs) ** 2
        total_ort_mse += np.sum(syy)
        ort_mse = np.mean(syy)

        absy = np.abs(to_numpy(y) - ort_outs)
        total_ort_mae += np.sum(absy)
        ort_mae = np.mean(absy)

        if use_cuda:
            x = x.to('cuda')
            y = y.to('cuda')

        with torch.no_grad():
            pt_outs = model(x)

        # Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE) for PyTorch outputs
        syy = (pt_outs - y) ** 2
        total_pt_mse += torch.sum(syy)
        pt_mse = torch.mean(syy).item()

        absy = torch.abs(pt_outs - y)
        total_pt_mae += torch.sum(absy)
        pt_mae = torch.mean(absy).item()

        # print(f"ONNX    MSE: {ort_mse:.6f}, MAE: {ort_mae:.6f}")
        # print(f"PyTorch MSE: {pt_mse:.6f}, MAE: {pt_mae:.6f}")

    N = len(dataset)
    print(f"pt   MSE = {total_pt_mse.item() / N:.6f}, MAE = {total_pt_mae.item() / N:.6f}")
    print(f"onnx MSE = {total_ort_mse / N:.6f}, MAE = {total_ort_mae.item() / N:.6f} ")
