# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import AutoEncoder

def train_autoencoder(train_dataset, input_dim=784, latent_dim=64, lr=1e-3, num_epochs=20, batch_size=64, device='cpu'):
    model = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch.view(batch.size(0), -1).to(device)
            output = model(x)
            loss = criterion(output, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[{epoch+1}/{num_epochs}] Loss: {total_loss/len(dataloader):.4f}")

    return model
