from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import onnxruntime as ort


def to_numpy(tensor):
    """ This function will allow us to use the same PyTorch DataLoader with ONNX. """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def test_models(
        onnx_model_fp32_path,
        quantized_model_path,
        ort_provider,
        dataset,
        batch_size,
):
    # Let's compare both models: FP32 vs. INT8
    ort_sess = ort.InferenceSession(onnx_model_fp32_path, providers=ort_provider)
    ort_int8_sess = ort.InferenceSession(quantized_model_path, providers=ort_provider)

    # create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_fp32_mse = 0
    total_fp32_mae = 0
    total_int8_mse = 0
    total_int8_mae = 0
    for x, y in tqdm(dataloader, ascii=True, unit="batches"):

        ort_inputs = {ort_sess.get_inputs()[0].name: to_numpy(x)}
        ort_outs = ort_sess.run(None, ort_inputs)[0]

        # Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE)
        syy = (to_numpy(y) - ort_outs) ** 2
        total_fp32_mse += np.sum(syy)
        fp32_mse = np.mean(syy)

        absy = np.abs(to_numpy(y) - ort_outs)
        total_fp32_mae += np.sum(absy)
        fp32_mae = np.mean(absy)

        ort_int8_outs = ort_int8_sess.run(None, ort_inputs)[0]

              # Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE) for PyTorch outputs
        syy = (ort_int8_outs - to_numpy(y)) ** 2
        total_int8_mse += np.sum(syy)
        int8_mse = np.mean(syy).item()

        absy = np.abs(ort_int8_outs - to_numpy(y))
        total_int8_mae += np.sum(absy)
        int8_mae = np.mean(absy).item()

        # print(f"FP32 MSE: {ort_mse:.6f}, MAE: {ort_mae:.6f}")
        # print(f"INT8 MSE: {pt_mse:.6f}, MAE: {pt_mae:.6f}")

    N = len(dataset)
    print(f"FP32 MSE = {total_fp32_mse / N:.6f}, MAE = {total_fp32_mae.item() / N:.6f} ")
    print(f"INT8 MSE = {total_int8_mse.item() / N:.6f}, MAE = {total_int8_mae.item() / N:.6f}")
