import os
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.quantization

from config import n_features, n_outputs
from config import batch_size, model_out
from config import best_model_name, pt_model_int8_name, onnx_model_int8_name
from dataset import MyRegressionDataset
from model import RegressionModel



if __name__ == "__main__":
    # create dataset
    dataset = MyRegressionDataset(n_features, n_outputs)
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = RegressionModel(n_features, n_outputs)
    # Load the best model's weights
    model.load_state_dict(torch.load(os.path.join(model_out, best_model_name)))
    model.train()

    # Define the quantization configuration
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')  # Or other backend

    # Prepare the model for QAT (inserts observers and fake quantization nodes)
    model_prepared = torch.quantization.prepare_qat(model)

    # Training loop (with QAT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_prepared.to(device) # Move model to device

    # Example training loop (adapt to your needs)
    num_epochs = 10  # Adjust as needed
    optimizer = torch.optim.Adam(model_prepared.parameters(), lr=0.001) # Example optimizer
    loss_fn = nn.CrossEntropyLoss() # Example loss function

    for epoch in tqdm(range(num_epochs)):
        for x, y in dataloader:
            x = x.to(device) # Move data to device
            y = y.to(device) # Move data to device
            optimizer.zero_grad()
            outputs = model_prepared(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

    # Save the quantized PyTorch model
    torch.save(model_prepared.state_dict(), os.path.join(model_out, pt_model_int8_name))
    print(f"Quantized PyTorch model saved to: {pt_model_int8_name}")

    # Convert to ONNX
    x, _ = dataset[0]
    dummy_input = x.to(device)  # Example dummy input size, adapt as needed. Move to device!
    model_prepared.eval()

    # # De-fuse the quantization
    # model_defused = torch.quantization.quantize_dynamic(
    #     model_prepared,
    #     {''},  # The empty set means all modules, change to a list of modules if you want to be more specific.
    # )

    # manual de-fuse
    def manual_dequantize(module):
        for name, child in module.named_children():
            if isinstance(child, torch.quantization.observer.MovingAverageMinMaxObserver) or isinstance(child, torch.quantization.observer.MinMaxObserver):
                # Replace observer with identity
                setattr(module, name, nn.Identity())
            elif isinstance(child, torch.quantization.fake_quantize.FakeQuantize):
                # Replace fake quantize with identity
                setattr(module, name, nn.Identity())
            else:
                manual_dequantize(child) # Recurse
        return module

    model_defused = manual_dequantize(model_prepared)

    torch.onnx.export(
        model_defused,  # Use the de-fused model
        dummy_input,
        os.path.join(model_out, onnx_model_int8_name),
        export_params=True,  # Export model weights
        # opset_version=19,  # Or your desired opset version
        input_names=["input"],  # Replace with your input names
        output_names=["output"],  # Replace with your output names
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # Optional, for dynamic batch size
    )

    print(f"Quantized ONNX model saved to: {onnx_model_int8_name}")
    print(model_defused)
