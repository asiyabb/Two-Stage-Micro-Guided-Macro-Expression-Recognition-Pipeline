import torch
import torch.nn as nn
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# --- 1. Exact Baseline LSTM Model Architecture ---
class BaselineLSTMModel(nn.Module):
    def __init__(self): 
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1050, 
            hidden_size=256, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.3, 
            bidirectional=True
        )
        self.linear = nn.Linear(512, 6)  # 6 Macro Expression classes
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return x    

# --- 2. Dataset Wrapper for 1050d Combined Features ---
class BaselineDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def evaluate_macro_baseline():
    print("--- Evaluating Macro-Only Baseline (1050d Features + LSTM) ---")
    
    # Path to the combined 1050d JSON file and baseline checkpoint
    json_path = "../MMEW_Macro_Pipeline(Baseline)/Combined_MMEW_1050d.json"
    checkpoint_path = "../MMEW_Macro_Pipeline(Baseline)/best_model_mmew.pth"

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[Error] Could not find '{json_path}'. Ensure paths are correct relative to your current directory.")
        return

    X = data['features']
    y = data['labels']

    # Use the exact same random state and split ratio as Training_LSTM_MMEW.py
    total_samples = len(X)
    test_size = int(total_samples * 0.2)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=50, stratify=y
    )

    test_data = BaselineDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BaselineLSTMModel().to(device)
    
    try:
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded baseline checkpoint from {checkpoint_path}")
    except Exception as e:
        print(f"[Error] Failed to load checkpoint: {e}")
        return

    net.eval()
    preds, targets = [], []
    
    with torch.no_grad():
        for x, t in test_loader:
            x, t = x.to(device), t.to(device)
            outputs = net(x)
            predicted = torch.argmax(outputs, dim=1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(t.cpu().numpy())

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')
    
    print("\n================ BASELINE RESULTS ================")
    print(f"Baseline Test Accuracy : {acc * 100:.2f}%")
    print(f"Baseline Macro F1      : {f1:.4f}")
    print("==================================================")
    
    classes_names = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
    print("\nClassification Report:")
    print(classification_report(targets, preds, target_names=classes_names, digits=4, zero_division=0))

if __name__ == "__main__":
    evaluate_macro_baseline()