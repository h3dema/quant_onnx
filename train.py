import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from config import batch_size, n_epochs, LR, step_size, gamma
from config import n_features, n_outputs
from config import model_out, best_model_name
from dataset import MyRegressionDataset
from model import RegressionModel


if __name__ == "__main__":

    # create dataset
    dataset = MyRegressionDataset(n_features, n_outputs)
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = RegressionModel(n_features, n_outputs)

    # Initialize loss function, and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Train model
    os.makedirs(model_out, exist_ok=True)
    best_loss = None
    for epoch in range(n_epochs):
        # Train mode
        model.train()
        total_loss = 0
        for inputs, targets in tqdm(dataloader):
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

        # Save best model based on total loss
        if epoch == 0 or total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), os.path.join(model_out, best_model_name))

        # Update learning rate
        scheduler.step()

        # Print loss at each epoch
        print(f'Epoch {epoch+1}, Loss: {total_loss / batch_size}')

        # Evaluate model (optional)
        # model.eval()
        # with torch.no_grad():
        #     # Evaluate model on validation set
        #     pass