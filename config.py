# Hyperparameters

n_features = 10
n_outputs = 2

n_epochs = 100
batch_size = 32
learning_rate = 0.01

LR = 0.01
step_size = 20
gamma = 0.1


# Configuration

# directory to save the model's weight
model_out = "model"
best_model_name = 'best_model.pth'

onnx_model_name = 'model.onnx'  # name of `best_model_name` conveerted to ONNX format
model_quant_name = "model_quantized.onnx"
pt_model_int8_name = "pt_model_int8.pth"
onnx_model_int8_name = "onnx_model_int8.onnx"
