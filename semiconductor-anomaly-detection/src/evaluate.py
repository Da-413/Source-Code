# evaluate.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_reconstruction_error(model, dataloader, device='cpu'):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch.view(batch.size(0), -1).to(device)
            output = model(x)
            loss = torch.mean((output - x) ** 2, dim=1)
            errors.extend(loss.cpu().numpy())
    return np.array(errors)

def evaluate(model, dataloader, labels, threshold, device='cpu'):
    errors = compute_reconstruction_error(model, dataloader, device)
    preds = (errors > threshold).astype(int)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    return errors, preds

def plot_error_distribution(errors, threshold):
    plt.hist(errors, bins=50, alpha=0.7, label='Reconstruction Error')
    plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.legend()
    plt.show()
