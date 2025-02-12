"""
ONNX does not support QAT anymore.

"""
import os
import argparse
import torch
import onnx
import onnxruntime as ort
from onnxruntime import quantization
from onnxruntime.quantization import QuantizationMode

from config import model_out, onnx_model_name, model_quant_name
from config import n_features, n_outputs
from config import batch_size
from dataset import MyRegressionDataset
from onnx_utils import test_models


def main():
    # path to the ONNX model
    onnx_model_fp32_path = os.path.join(model_out, onnx_model_name)
    # where to save the quantized model
    quantized_model_path = os.path.join(model_out, model_quant_name)

    # do quantization
    quantized_model = ort.quantize_dynamic(
        onnx_model_fp32_path,
        quantized_model_path,
        weight_type=ort.quantize.QType.QUInt8
    )


class QuantizationDataReader(quantization.CalibrationDataReader):
    def __init__(self, torch_ds, batch_size, input_name):

        self.torch_dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, shuffle=False)

        self.input_name = input_name
        self.datasize = len(self.torch_dl)

        self.enum_data = iter(self.torch_dl)

    def to_numpy(self, pt_tensor):
        return pt_tensor.detach().cpu().numpy() if pt_tensor.requires_grad else pt_tensor.cpu().numpy()

    def get_next(self):
        batch = next(self.enum_data, None)
        if batch is not None:
          return {self.input_name: self.to_numpy(batch[0])}
        else:
          return None

    def rewind(self):
        self.enum_data = iter(self.torch_dl)


def menu():
    parser = argparse.ArgumentParser(description='ONNX Quantization Script')
    parser.add_argument('--use-static', dest="use_static", action='store_true', help='Enable static quantization')
    parser.add_argument('--no-use-static', dest="use_static", action='store_false', help='Disable static quantization')
    parser.set_defaults(use_static=False)

    parser.add_argument('--use-dynamic', dest="use_dynamic", action='store_true', help='Enable dynamic quantization')
    parser.add_argument('--no-use-dynamic', dest="use_dynamic", action='store_false', help='Disable dynamic quantization')
    parser.set_defaults(use_dynamic=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = menu()

    # create dataset
    dataset = MyRegressionDataset(n_features, n_outputs)

    # path to the ONNX model
    onnx_model_fp32_path = os.path.join(model_out, onnx_model_name)
    # where to save the quantized model
    quantized_model_path = os.path.join(model_out, model_quant_name)

    # load the model
    model_onnx = onnx.load(onnx_model_fp32_path)
    onnx.checker.check_model(model_onnx)

    # create Inference session,
    # defining the provider and quantization options depending on GPU availability
    ort_provider = ['CPUExecutionProvider']
    q_static_opts = {"ActivationSymmetric":False,
                    "WeightSymmetric":True}
    use_cuda = False  #  torch.cuda.is_available()
    if use_cuda:
         # will require cuDNN 9.* and CUDA 12.*.
        ort_provider = ['CUDAExecutionProvider']
        q_static_opts = {"ActivationSymmetric":True,
                        "WeightSymmetric":True}
    ort_sess = ort.InferenceSession(onnx_model_fp32_path, providers=ort_provider)

    # Preprocessing step:
    # -------------------
    # includes optimizations
    # it is recommended to be performed prior to quantization,
    # according to ONNX Runtime Documentation.
    model_prep_path = quantized_model_path.replace("_quantized", "_prep")
    quantization.shape_inference.quant_pre_process(
        onnx_model_fp32_path,
        model_prep_path,
        skip_symbolic_shape=False)

    qdr = QuantizationDataReader(
        dataset,
        batch_size=batch_size,
        input_name=ort_sess.get_inputs()[0].name
    )

    if args.use_static:
        print("Static quantization")
        quantized_model = quantization.quantize_static(
            model_input=model_prep_path,
            model_output=quantized_model_path,
            calibration_data_reader=qdr,
            extra_options=q_static_opts
        )
        test_models(
            onnx_model_fp32_path,
            quantized_model_path,
            ort_provider,
            dataset,
            batch_size,
        )

    if args.use_dynamic:
        print("Dynamic quantization")
        dyn_quantized_model_path = quantized_model_path.replace("_quantized", "_dyn_quantized")
        quantized_model = quantization.quantize_dynamic(
            model_input=model_prep_path,
            model_output=dyn_quantized_model_path,
            extra_options={"EnableSubgraph": True}  # whether subgraph will be quantized.
        )
        test_models(
            onnx_model_fp32_path,
            dyn_quantized_model_path,
            ort_provider,
            dataset,
            batch_size,
        )
