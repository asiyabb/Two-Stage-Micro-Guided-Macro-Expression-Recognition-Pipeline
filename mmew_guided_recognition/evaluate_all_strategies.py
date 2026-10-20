import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from config.config import Config
from data.dataset import MMEWSequenceDataset, get_subject_splits
from models.vit_extractor import ViTFeatureExtractor
from models.micro_encoder import MotionMicroEncoder
from models.macro_guided_model import MacroGuidedModel
from losses.micro_losses import ContrastiveWindowLoss, TemporalConsistencyLoss


# --- 1. Baseline Model Architecture & Dataset ---
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
        self.linear = nn.Linear(512, 6)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return x    

class BaselineDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def evaluate_baseline_metrics():
    """Evaluates the macro-only baseline using its 1050d combined features."""
    json_path = "../MMEW_Macro_Pipeline(Baseline)/Combined_MMEW_1050d.json"
    checkpoint_path = "../MMEW_Macro_Pipeline(Baseline)/best_model_mmew.pth"

    if not os.path.exists(json_path) or not os.path.exists(checkpoint_path):
        print("[Warning] Baseline files not found. Skipping baseline evaluation.")
        return None, None

    with open(json_path, 'r') as f:
        data = json.load(f)

    X = data['features']
    y = data['labels']
    test_size = int(len(X) * 0.2)
    _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=50, stratify=y)

    test_loader = DataLoader(BaselineDataset(X_test, y_test), batch_size=16)
    net = BaselineLSTMModel().to(Config.DEVICE)
    net.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
    net.eval()

    preds, targets = [], []
    with torch.no_grad():
        for x, t in test_loader:
            x, t = x.to(Config.DEVICE), t.to(Config.DEVICE)
            outputs = net(x)
            predicted = torch.argmax(outputs, dim=1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(t.cpu().numpy())

    return accuracy_score(targets, preds), f1_score(targets, preds, average='macro')


def run_strategy_1(train_loader, val_loader, vit):
    """Strategy 1: Pretrain micro-encoder, freeze it, train full macro-guided model."""
    print("\n" + "="*60)
    print("RUNNING STRATEGY 1: Two-Stage (Pretrain Micro -> Freeze -> Train Macro)")
    print("="*60)

    micro_dataset = MMEWSequenceDataset(mode="micro")
    micro_loader = DataLoader(micro_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    micro_encoder = MotionMicroEncoder().to(Config.DEVICE)
    contrastive_criterion = ContrastiveWindowLoss()
    temporal_criterion = TemporalConsistencyLoss()
    optimizer_micro = torch.optim.Adam(micro_encoder.parameters(), lr=Config.LR)

    strat1_micro_losses = []
    micro_encoder.train()

    print("\n[Strategy 1 - Step A] Pretraining Micro Encoder...")
    for epoch in range(Config.EPOCHS_PRETRAIN):
        total_loss = 0.0
        for x, _ in micro_loader:
            x = x.to(Config.DEVICE)
            with torch.no_grad():
                vit_feats = vit(x)

            latent_clues = micro_encoder(vit_feats)
            loss = (Config.ALPHA_CONTRASTIVE * contrastive_criterion(latent_clues) +
                    Config.BETA_TEMPORAL * temporal_criterion(latent_clues))

            optimizer_micro.zero_grad()
            loss.backward()
            optimizer_micro.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(micro_loader)
        strat1_micro_losses.append(avg_loss)
        print(f"Epoch [{epoch+1:02d}/{Config.EPOCHS_PRETRAIN}] | Micro Pretrain Loss: {avg_loss:.4f}")

    print("\n[Strategy 1 - Step B] Training Full Model with Micro-Encoder Frozen...")
    for param in micro_encoder.parameters():
        param.requires_grad = False
    micro_encoder.eval()

    guided_model_s1 = MacroGuidedModel(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer_macro = torch.optim.Adam(guided_model_s1.parameters(), lr=Config.LR)

    s1_train_losses, s1_val_losses = [], []
    s1_train_accs, s1_val_accs = [], []
    best_s1_acc = 0.0

    for epoch in range(Config.EPOCHS_MACRO):
        guided_model_s1.train()
        train_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            with torch.no_grad():
                vit_feats = vit(x)
                latent_clues = micro_encoder(vit_feats)

            logits = guided_model_s1(vit_feats, latent_clues)
            loss = criterion(logits, y)

            optimizer_macro.zero_grad()
            loss.backward()
            optimizer_macro.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        t_acc = correct / total
        t_loss = train_loss / len(train_loader)

        guided_model_s1.eval()
        val_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                vit_feats = vit(x)
                latent_clues = micro_encoder(vit_feats)
                logits = guided_model_s1(vit_feats, latent_clues)
                loss = criterion(logits, y)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                v_correct += (preds == y).sum().item()
                v_total += y.size(0)

        v_acc = v_correct / v_total
        v_loss = val_loss / len(val_loader)

        s1_train_losses.append(t_loss)
        s1_val_losses.append(v_loss)
        s1_train_accs.append(t_acc)
        s1_val_accs.append(v_acc)

        print(f"Epoch [{epoch+1:02d}/{Config.EPOCHS_MACRO}] | Train Loss: {t_loss:.4f} | Train Acc: {t_acc*100:.2f}% | Val Loss: {v_loss:.4f} | Val Acc: {v_acc*100:.2f}%")

        if v_acc > best_s1_acc:
            best_s1_acc = v_acc
            torch.save(guided_model_s1.state_dict(), "guided_model_strategy1_best.pth")
            torch.save(micro_encoder.state_dict(), "micro_encoder_strategy1_best.pth")

    return {
        "micro_pretrain_loss": strat1_micro_losses,
        "train_loss": s1_train_losses,
        "val_loss": s1_val_losses,
        "train_acc": s1_train_accs,
        "val_acc": s1_val_accs
    }


def run_strategy_2(train_loader, val_loader, vit):
    """Strategy 2: Jointly train micro-encoder and full model end-to-end."""
    print("\n" + "="*60)
    print("RUNNING STRATEGY 2: Joint End-to-End Full Model Training")
    print("="*60)

    micro_encoder_s2 = MotionMicroEncoder().to(Config.DEVICE)
    guided_model_s2 = MacroGuidedModel(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(micro_encoder_s2.parameters()) + list(guided_model_s2.parameters()), 
        lr=Config.LR
    )

    s2_train_losses, s2_val_losses = [], []
    s2_train_accs, s2_val_accs = [], []
    best_s2_acc = 0.0

    for epoch in range(Config.EPOCHS_MACRO):
        micro_encoder_s2.train()
        guided_model_s2.train()
        train_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            with torch.no_grad():
                vit_feats = vit(x)

            latent_clues = micro_encoder_s2(vit_feats)
            logits = guided_model_s2(vit_feats, latent_clues)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        t_acc = correct / total
        t_loss = train_loss / len(train_loader)

        micro_encoder_s2.eval()
        guided_model_s2.eval()
        val_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                vit_feats = vit(x)
                latent_clues = micro_encoder_s2(vit_feats)
                logits = guided_model_s2(vit_feats, latent_clues)
                loss = criterion(logits, y)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                v_correct += (preds == y).sum().item()
                v_total += y.size(0)

        v_acc = v_correct / v_total
        v_loss = val_loss / len(val_loader)

        s2_train_losses.append(t_loss)
        s2_val_losses.append(v_loss)
        s2_train_accs.append(t_acc)
        s2_val_accs.append(v_acc)

        print(f"Epoch [{epoch+1:02d}/{Config.EPOCHS_MACRO}] | Train Loss: {t_loss:.4f} | Train Acc: {t_acc*100:.2f}% | Val Loss: {v_loss:.4f} | Val Acc: {v_acc*100:.2f}%")

        if v_acc > best_s2_acc:
            best_s2_acc = v_acc
            torch.save(guided_model_s2.state_dict(), "guided_model_strategy2_best.pth")
            torch.save(micro_encoder_s2.state_dict(), "micro_encoder_strategy2_best.pth")

    return {
        "train_loss": s2_train_losses,
        "val_loss": s2_val_losses,
        "train_acc": s2_train_accs,
        "val_acc": s2_val_accs
    }


def plot_curves(history_s1, history_s2):
    """Generates and saves train/test loss and accuracy curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history_s1["micro_pretrain_loss"], label="Pretrain Loss", color="purple")
    axes[0, 0].set_title("Strategy 1: Micro-Encoder Pretrain Loss Curve")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Contrastive + Temporal Loss")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(history_s1["train_acc"], label="Train Acc", color="blue")
    axes[0, 1].plot(history_s1["val_acc"], label="Val Acc", color="orange")
    axes[0, 1].set_title("Strategy 1: Full Model (Frozen Micro Branch)")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].plot(history_s2["train_loss"], label="Train Loss", color="blue", linestyle="--")
    axes[1, 0].plot(history_s2["val_loss"], label="Val Loss", color="red", linestyle="--")
    axes[1, 0].set_title("Strategy 2: Joint End-to-End Loss Curves")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].plot(history_s2["train_acc"], label="Train Acc", color="green")
    axes[1, 1].plot(history_s2["val_acc"], label="Val Acc", color="red")
    axes[1, 1].set_title("Strategy 2: Joint End-to-End Accuracy Curves")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("strategy_curves_comparison.png", dpi=300)
    print("\n[Info] Training curves saved to 'strategy_curves_comparison.png'")


def evaluate_comparative_summary(val_loader, vit):
    """Evaluates Strategy 1 vs Strategy 2 vs Baseline."""
    print("\n" + "="*65)
    print("FINAL EVALUATION: Strategy 1 vs Strategy 2 vs Macro-Only Baseline")
    print("="*65)

    results = {}

    # --- 1. Evaluate Baseline Model ---
    b_acc, b_f1 = evaluate_baseline_metrics()
    if b_acc is not None:
        results["Baseline (ShuffleNet + LSTM 1050d)"] = {"Accuracy": b_acc, "Macro_F1": b_f1}

    # --- 2. Evaluate Strategy 1 Model ---
    if os.path.exists("guided_model_strategy1_best.pth") and os.path.exists("micro_encoder_strategy1_best.pth"):
        model_s1 = MacroGuidedModel(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
        micro_s1 = MotionMicroEncoder().to(Config.DEVICE)

        model_s1.load_state_dict(torch.load("guided_model_strategy1_best.pth", map_location=Config.DEVICE))
        micro_s1.load_state_dict(torch.load("micro_encoder_strategy1_best.pth", map_location=Config.DEVICE))
        model_s1.eval()
        micro_s1.eval()

        preds, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                vit_feats = vit(x)
                latent_clues = micro_s1(vit_feats)
                logits = model_s1(vit_feats, latent_clues)
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                targets.extend(y.cpu().numpy())

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average="macro")
        results["Strategy 1 (Frozen Micro Branch)"] = {"Accuracy": acc, "Macro_F1": f1}

    # --- 3. Evaluate Strategy 2 Model ---
    if os.path.exists("guided_model_strategy2_best.pth") and os.path.exists("micro_encoder_strategy2_best.pth"):
        model_s2 = MacroGuidedModel(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
        micro_s2 = MotionMicroEncoder().to(Config.DEVICE)

        model_s2.load_state_dict(torch.load("guided_model_strategy2_best.pth", map_location=Config.DEVICE))
        micro_s2.load_state_dict(torch.load("micro_encoder_strategy2_best.pth", map_location=Config.DEVICE))
        model_s2.eval()
        micro_s2.eval()

        preds, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                vit_feats = vit(x)
                latent_clues = micro_s2(vit_feats)
                logits = model_s2(vit_feats, latent_clues)
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                targets.extend(y.cpu().numpy())

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average="macro")
        results["Strategy 2 (Joint End-to-End)"] = {"Accuracy": acc, "Macro_F1": f1}

    # --- Display Comparison Table ---
    print("\n" + "="*65)
    print(f"{'Model / Strategy':<35} | {'Accuracy':<10} | {'Macro F1':<10}")
    print("="*65)
    for model_name, metrics in results.items():
        print(f"{model_name:<35} | {metrics['Accuracy']*100:6.2f}%    | {metrics['Macro_F1']:8.4f}")
    print("="*65)


if __name__ == "__main__":
    dataset = MMEWSequenceDataset(mode="macro")
    train_set, val_set = get_subject_splits(dataset, test_size=0.15, seed=42)

    train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=Config.BATCH_SIZE, shuffle=False)

    vit = ViTFeatureExtractor(freeze=True).to(Config.DEVICE)
    vit.eval()

    history_s1 = run_strategy_1(train_loader, val_loader, vit)
    history_s2 = run_strategy_2(train_loader, val_loader, vit)

    plot_curves(history_s1, history_s2)
    evaluate_comparative_summary(val_loader, vit)