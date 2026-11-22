import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from scipy.stats import norm
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

def calculate_dsr_for_auc(auc_score, n_samples, num_trials=10):
    """
    DSR adaptation for AUC.
    Using the standard error AUC and multiple testing.
    """
    if n_samples == 0: return 0.0
    # Standard error AUC (Hanley & McNeil, 1982)
    std_error = np.sqrt(auc_score * (1 - auc_score) / n_samples)
    if std_error == 0: return 0.0
    
    # Awaited maximum AUC with random selection (N tests)
    gamma = 0.5772156649
    e_max_auc = 0.5 + np.sqrt(1 / n_samples) * (
        (1 - gamma) * norm.ppf(1 - 1/num_trials) + 
        gamma * norm.ppf(1 - 1/(num_trials * np.e))
    )
    
    z_score = (auc_score - e_max_auc) / std_error
    dsr = norm.cdf(z_score)
    return dsr

def train_toxicity_detector():
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_parquet(DATASET_PATH)
    
    features = ['queue_imbalance', 'ofi', 'microprice_dev', 'vpin_approx']
    target = 'is_toxic'
    
    df_model = df.dropna(subset=features + [target]).copy()
    
    print(f"Total valid trades: {len(df_model):,}")
    print(f"Toxic rate: {df_model[target].mean()*100:.2f}%")
    
    # TIME-BASED SPLIT
    split_idx = int(len(df_model) * 0.8)
    
    train_df = df_model.iloc[:split_idx]
    test_df = df_model.iloc[split_idx:]
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")
    
    # LIGHTGBM training
    print("\nTraining LightGBM Detector...")
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    # Qualitative evaluation (AUC-ROC & Precision/Recall)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n DETECTOR PERFORMANCE (OOS)")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Statistical importance (DSR)
    n_test_samples = len(y_test)
    dsr_auc = calculate_dsr_for_auc(auc, n_test_samples, num_trials=10)
    print(f"Deflated AUC (DSR): {dsr_auc:.4f} (Threshold: > 0.95)")
    
    # Execution Filter
    threshold = 0.6
    filtered_trades = test_df[y_pred_proba < threshold]
    
    original_toxic_rate = test_df[target].mean() * 100
    filtered_toxic_rate = filtered_trades[target].mean() * 100
    
    print("\n Execution Simulation (Filter threshold = 0.6)")
    print(f"Original Toxic Rate in Test: {original_toxic_rate:.2f}%")
    print(f"Filtered Toxic Rate: {filtered_toxic_rate:.2f}%")
    print(f"Reduction in Toxicity: {original_toxic_rate - filtered_toxic_rate:.2f}%")
    print(f"Amount of trades we avoided: {len(test_df) - len(filtered_trades):,} ({(1 - len(filtered_trades)/len(test_df))*100:.1f}%)")

    plt.figure(figsize=(10, 5))
    lgb.plot_importance(model, max_num_features=10, importance_type='gain')
    plt.title('Feature Importance (LightGBM)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png")
    print(f"\nFeature importance plot saved to {PLOTS_DIR / 'feature_importance.png'}")
    plt.close()

if __name__ == "__main__":
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}. Run toxicity_features.py first.")
    else:
        train_toxicity_detector()