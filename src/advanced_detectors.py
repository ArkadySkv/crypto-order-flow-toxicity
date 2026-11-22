import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
PLOTS_DIR = PROJECT_ROOT / "results/plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "BTCUSDT"
DATE = "2024-01-02"
DATASET_PATH = PROCESSED_DIR / f"{SYMBOL}_toxicity_features_{DATE}.parquet"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading dataset...")
df = pd.read_parquet(DATASET_PATH)
features = ['queue_imbalance', 'ofi', 'microprice_dev', 'vpin_approx']
target = 'is_toxic'
df = df.dropna(subset=features + [target]).copy()

print("Applying winsorization to handle heavy tails...")
for col in features:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = np.clip(df[col], lower, upper)

# Time-based split
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train_raw = train_df[features].values
y_train = train_df[target].values
X_test_raw = test_df[features].values
y_test = test_df[target].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

X_train_t = torch.FloatTensor(X_train).to(DEVICE)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
X_test_t = torch.FloatTensor(X_test).to(DEVICE)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

# Weights to class disbalance
pos_weight_val = (y_train == 0).sum() / (y_train == 1).sum()
pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(DEVICE)

# BASELINE: LightGBM
print("\n Training Baseline: LightGBM ")
lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, scale_pos_weight=pos_weight_val, random_state=42, n_jobs=-1, verbose=-1)
lgb_model.fit(X_train, y_train)
lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
lgb_auc = roc_auc_score(y_test, lgb_probs)
print(f"LightGBM AUC: {lgb_auc:.4f}")

# NEURO-SYMBOLIC: Logic Tensor Networks (LTN)
print("\n Training Neuro-Symbolic: LTN ")

class LTNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32), nn.BatchNorm1d(32), nn.LeakyReLU(),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.LeakyReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

model_ltn = LTNNet().to(DEVICE)
optimizer_ltn = optim.Adam(model_ltn.parameters(), lr=0.001)
criterion_ltn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

for epoch in range(15):
    model_ltn.train()
    optimizer_ltn.zero_grad()
    
    logits = model_ltn(X_train_t)
    bce_loss = criterion_ltn(logits, y_train_t)
    
    # Simplier logic: If OFI > 1 (higher), the probability of toxicity should be > 0.5
    # We are using little penalty to prevent gradient dead
    ofi_tensor = X_train_t[:, 1].unsqueeze(1)
    preds = torch.sigmoid(logits)
    # Penalty: OFI > 1, preds < 0.5
    logic_penalty = torch.mean(torch.relu(0.5 - preds) * torch.relu(ofi_tensor - 1.0))
    
    total_loss = bce_loss + 0.1 * logic_penalty
    total_loss.backward()
    optimizer_ltn.step()
    
    if (epoch+1) % 5 == 0:
        print(f"  Epoch {epoch+1}/15 | BCE: {bce_loss.item():.4f} | Logic: {logic_penalty.item():.4f}")

model_ltn.eval()
with torch.no_grad():
    ltn_probs = torch.sigmoid(model_ltn(X_test_t)).cpu().numpy().flatten()
ltn_auc = roc_auc_score(y_test, ltn_probs)
print(f"LTN AUC: {ltn_auc:.4f}")

# PINN: Physics-Informed Neural Network
print("\n Training PINN: Physics-Informed Network ")

class PINNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.BatchNorm1d(64), nn.SiLU(),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model_pinn = PINNNet().to(DEVICE)
optimizer_pinn = optim.Adam(model_pinn.parameters(), lr=0.0005)
criterion_pinn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

X_train_pinn = X_train_t.clone().detach().requires_grad_(True)

for epoch in range(20):
    model_pinn.train()
    optimizer_pinn.zero_grad()
    
    logits = model_pinn(X_train_pinn)
    bce_loss = criterion_pinn(logits, y_train_t)
    
    # Physics: OFI (gradient >= 0)
    preds = torch.sigmoid(logits)
    grad_ofi = torch.autograd.grad(preds.sum(), X_train_pinn, create_graph=True)[0][:, 1]
    
    physics_loss = torch.mean(torch.relu(-grad_ofi))
    
    total_loss = bce_loss + 0.01 * physics_loss
    total_loss.backward()
    optimizer_pinn.step()
    
    if (epoch+1) % 5 == 0:
        print(f"  Epoch {epoch+1}/20 | BCE: {bce_loss.item():.4f} | Physics: {physics_loss.item():.4f}")

model_pinn.eval()
with torch.no_grad():
    pinn_probs = torch.sigmoid(model_pinn(X_test_t)).cpu().numpy().flatten()
pinn_auc = roc_auc_score(y_test, pinn_probs)
print(f"PINN AUC: {pinn_auc:.4f}")

# PULSE : Bayesian Last Layer
print("\n Training PULSE: Bayesian Last Layer (arXiv:2312.05827) ")

# Use PINN architecture, but freezing all layers and training the last one
model_pulse = PINNNet().to(DEVICE)
model_pulse.load_state_dict(model_pinn.state_dict())

for param in model_pulse.parameters():
    param.requires_grad = False

# Unfreeze only the last layer
for param in model_pulse.net[-1].parameters():
    param.requires_grad = True

optimizer_pulse = optim.Adam(model_pulse.net[-1].parameters(), lr=0.005)
criterion_pulse = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

for epoch in range(30):
    model_pulse.train()
    optimizer_pulse.zero_grad()
    logits = model_pulse(X_train_t)
    loss = criterion_pulse(logits, y_train_t)
    loss.backward()
    optimizer_pulse.step()

model_pulse.eval()
with torch.no_grad():
    pulse_probs = torch.sigmoid(model_pulse(X_test_t)).cpu().numpy().flatten()
pulse_auc = roc_auc_score(y_test, pulse_probs)
print(f"PULSE (Bayesian Last Layer) AUC: {pulse_auc:.4f}")

print("\n COMPARISON")
print(f"LightGBM (Baseline): {lgb_auc:.4f}")
print(f"LTN (Neuro-Symbolic): {ltn_auc:.4f}")
print(f"PINN (Physics-Informed): {pinn_auc:.4f}")
print(f"PULSE (Online Bayesian): {pulse_auc:.4f}")

plt.figure(figsize=(10, 6))
for name, probs in [("LightGBM", lgb_probs), ("LTN", ltn_probs), ("PINN", pinn_probs), ("PULSE", pulse_probs)]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, probs):.4f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: Advanced Detectors vs Baseline')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(PLOTS_DIR / "advanced_detectors_comparison.png", dpi=300)
print(f"\nComparison plot saved to {PLOTS_DIR / 'advanced_detectors_comparison.png'}")