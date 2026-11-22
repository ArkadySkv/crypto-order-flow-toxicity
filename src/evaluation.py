import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
PLOTS_DIR = PROJECT_ROOT / "results/plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "BTCUSDT"
DATE = "2024-01-02"
DATASET_PATH = PROCESSED_DIR / f"{SYMBOL}_toxicity_features_{DATE}.parquet"

def generate_paper_plots():
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_parquet(DATASET_PATH)
    
    features = ['queue_imbalance', 'ofi', 'microprice_dev', 'vpin_approx']
    target = 'is_toxic'
    df_model = df.dropna(subset=features + [target]).copy()
    
    # Time-based split
    split_idx = int(len(df_model) * 0.8)
    train_df = df_model.iloc[:split_idx]
    test_df = df_model.iloc[split_idx:]
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    print("Training model to extract predictions for plots...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, verbose=-1
    )
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # ROC CURVE
    print("Generating ROC Curve...")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Toxicity Detector')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(PLOTS_DIR / "roc_curve.png", dpi=300)
    plt.close()

    # CONFUSION MATRIX
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-Toxic (0)', 'Toxic (1)'],
                yticklabels=['Non-Toxic (0)', 'Toxic (1)'])
    plt.title('Confusion Matrix (Threshold = 0.5)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    # FEATURE DISTRIBUTIONS (Toxic vs Non-Toxic)
    print("Generating Feature Distributions...")
    _, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # OFI Distribution
    sns.kdeplot(data=test_df, x='ofi', hue=target, common_norm=False, 
                palette={0: "blue", 1: "red"}, fill=True, ax=axes[0], alpha=0.5)
    axes[0].set_title('Order Flow Imbalance (OFI) Distribution')
    axes[0].set_xlim(-50, 50)
    
    # Microprice Deviation Distribution
    sns.kdeplot(data=test_df, x='microprice_dev', hue=target, common_norm=False, 
                palette={0: "blue", 1: "red"}, fill=True, ax=axes[1], alpha=0.5)
    axes[1].set_title('Microprice Deviation Distribution')
    axes[1].set_xlim(-0.0005, 0.0005)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_distributions.png", dpi=300)
    plt.close()

    # TOXICITY REDUCTION BAR CHART
    print("Generating Toxicity Reduction Plot...")
    threshold = 0.6
    filtered_trades = test_df[y_pred_proba < threshold]
    
    original_toxic_rate = test_df[target].mean() * 100
    filtered_toxic_rate = filtered_trades[target].mean() * 100
    
    plt.figure(figsize=(7, 5))
    bars = plt.bar(['Original', 'Filtered'], [original_toxic_rate, filtered_toxic_rate], color=['tab:red', 'tab:green'])
    plt.ylabel('Toxicity Rate (%)')
    plt.title(f'Impact of Toxicity Filter (Threshold = {threshold})')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.2f}%', ha='center', va='bottom')
        
    plt.ylim(0, max(original_toxic_rate, filtered_toxic_rate) + 2)
    plt.savefig(PLOTS_DIR / "toxicity_reduction.png", dpi=300)
    plt.close()

    print("\n--- VERIFICATION ---")
    print(f"Looking for files in: {PLOTS_DIR.absolute()}")
    saved_files = os.listdir(PLOTS_DIR)
    print(f"Files in directory: {saved_files}")
    
    for f in ["roc_curve.png", "confusion_matrix.png", "feature_distributions.png", "toxicity_reduction.png"]:
        if f in saved_files:
            print(f"[OK] {f} saved successfully.")
        else:
            print(f"[ERROR] {f} NOT FOUND!")

if __name__ == "__main__":
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}. Run toxicity_features.py first.")
    else:
        generate_paper_plots()