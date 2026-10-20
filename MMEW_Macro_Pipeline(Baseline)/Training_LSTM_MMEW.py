# -*- coding: utf-8 -*-
"""
LSTM Training for MMEW Macro Expressions (With Image Exporters for CM and Report)
"""
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class lstm_model(nn.Module):
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
        self.linear = nn.Linear(512, 6)    # 6 Macro Expression classes
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:,-1,:])
        return x    

json_path = r"Combined_MMEW_1050d.json"

with open(json_path, 'r') as f:
  data = json.load(f)

class CreateDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

total_samples = len(data['features'])
test_size = int(total_samples * 0.2)
batch_size = 16

X_train, X_test, y_train, y_test = train_test_split(
    data['features'], data['labels'], test_size=test_size, random_state=50, stratify=data['labels']
)

train_data = CreateDataset(X_train, y_train)
test_data = CreateDataset(X_test, y_test)
Trainloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
Testloader = DataLoader(test_data, batch_size=batch_size)

# Instantiate network
net = lstm_model().cuda()
print(net)

loss_fnc = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0003, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

Num_epoch = 100
save_ruta = 'best_model_mmew.pth'
best_loss = float('inf')

for epoch in tqdm(range(Num_epoch)):
    net.train()
    loss_train = 0
    for j, (x, t) in enumerate(Trainloader):
        x, t = x.cuda(), t.cuda()
        y = net(x)
        loss = loss_fnc(y, t)
        loss_train += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        
    loss_train /= j+1

    net.eval()
    loss_test = 0
    for j, (x, t) in enumerate(Testloader):
        x, t = x.cuda(), t.cuda()
        y = net(x)
        loss = loss_fnc(y, t)
        loss_test += loss.item()
    loss_test /= j+1

    scheduler.step(loss_test)

    if loss_test < best_loss:
        torch.save(net.state_dict(), save_ruta)
        best_loss = loss_test

print(f"\n✅ Entrenamiento completado. Mejor modelo guardado en {save_ruta}")

# Evaluation and Image Export
net.load_state_dict(torch.load(save_ruta)) 
net.cuda()
net.eval()

Testloader_2 = DataLoader(test_data, batch_size=len(test_data))
for (x_all, t_all) in Testloader_2:
    x_all, t_all = x_all.cuda(), t_all.cuda()
    y_all = net(x_all)
    
y_all_label = y_all.argmax(1).cpu()
t_all = t_all.cpu()

classes_names = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

# 1. Save Confusion Matrix as Image
cm = confusion_matrix(t_all, y_all_label) 
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes_names, yticklabels=classes_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Matriz de Confusión - MMEW Macro Expressions")
plt.savefig('confusion_matrix_mmew.png', dpi=300, bbox_inches='tight')
plt.close()
print("💾 Confusion matrix saved as 'confusion_matrix_mmew.png'")

# 2. Save Classification Report as a Table Image
report_dict = classification_report(t_all, y_all_label, target_names=classes_names, digits=4, zero_division=0, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()

plt.figure(figsize=(9, 5))
plt.axis('off')
plt.title("Classification Report - MMEW Macro Expressions", fontsize=14, weight='bold', pad=20)
the_table = plt.table(cellText=df_report.round(4).values,
                      colLabels=df_report.columns,
                      rowLabels=df_report.index,
                      loc='center',
                      cellLoc='center')
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
the_table.scale(1.2, 1.5)

plt.savefig('classification_report_mmew.png', dpi=300, bbox_inches='tight')
plt.close()
print("💾 Classification report table saved as 'classification_report_mmew.png'")

# Print to Terminal
print("\n=== Classification Report ===")
print(classification_report(t_all, y_all_label, target_names=classes_names, digits=4, zero_division=0))

acc = accuracy_score(t_all, y_all_label)
prec = precision_score(t_all, y_all_label, average='macro', zero_division=0)
rec = recall_score(t_all, y_all_label, average='macro', zero_division=0)
f1 = f1_score(t_all, y_all_label, average='macro', zero_division=0)

print("=== Métricas Generales ===")
print(f"Accuracy: {acc:.4f}")
print(f"Precision (macro): {prec:.4f}")
print(f"Recall (macro): {rec:.4f}")
print(f"F1 Score (macro): {f1:.4f}\n")