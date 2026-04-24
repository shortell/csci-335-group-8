import pandas as pd
import numpy as np
import sys, os
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, fbeta_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight

# --- CONFIGURATION ---
INPUT  = r'data\final\musk_events_k10_replies_True.csv'
REPORT = r'xgboost_report_v2.txt'
TARGET = 'max_z_next5'
THRESH = 1.5
FEATURES = [
    'mentions_tesla', 'is_reply', 'is_quote', 'is_retweet',
    'positive', 'negative', 'neutral',
    'close_delta_z', 'volume_delta_z',
    'price_cv', 'volume_cv'
]

# --- DATA PREP ---
df = pd.read_csv(INPUT)
X_raw = SimpleImputer(strategy='median').fit_transform(df[FEATURES])
y_raw = df[TARGET].apply(lambda z: 'buy' if not pd.isna(z) and z > THRESH else ('dont_buy' if not pd.isna(z) else np.nan))
mask = y_raw.notna().values
X, y = X_raw[mask], np.array(y_raw[y_raw.notna()])

# 80/20 Split
X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = MinMaxScaler()
X_tv_scaled = scaler.fit_transform(X_tv)
X_test_scaled = scaler.transform(X_test)

le = LabelEncoder()
y_tv_enc = le.fit_transform(y_tv)
BUY_IDX = list(le.classes_).index('buy')

# --- HYPERPARAMETER SEARCH ---
# We focus on parameters that constrain complexity to mimic Logistic Regression's simplicity
param_dist = {
    'learning_rate': [0.005, 0.01, 0.02, 0.05], # Slower than previous 0.1/0.2 [cite: 1]
    'max_depth': [1, 2, 3],                    # "Stumps" to prevent overfitting 
    'n_estimators': [300, 500],                # More trees, but smaller steps
    'subsample': [0.5, 0.6, 0.7],              # More aggressive row sampling
    'colsample_bytree': [0.5, 0.6, 0.8],       # Feature sampling
    'min_child_weight': [5, 10, 20],           # Higher values prevent specific leaves 
    'gamma': [0.5, 1, 5],                      # Minimum loss reduction for a split
    'reg_alpha': [0.1, 1, 10],                 # L1 Regularization (Lasso-like)
    'reg_lambda': [1, 10, 100]                 # L2 Regularization (Ridge-like)
}

# Use Stratified K-Fold for more stable validation than a single split
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sw = compute_sample_weight('balanced', y=y_tv_enc)

clf_base = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)

# Randomized search is faster for laptops and avoids grid-lock
search = RandomizedSearchCV(
    clf_base, 
    param_distributions=param_dist, 
    n_iter=40,              # Total model fits: 40 * 5 folds = 200 fits
    scoring='balanced_accuracy', 
    cv=cv, 
    verbose=1, 
    random_state=42
)

print("Starting hyperparameter search...")
search.fit(X_tv_scaled, y_tv_enc, sample_weight=sw)

best_model = search.best_estimator_

# --- THRESHOLD TUNING ---
def predict_thresh(clf, Xs, t=0.5):
    return np.where(clf.predict_proba(Xs)[:, BUY_IDX] >= t, 'buy', 'dont_buy')

def tune_threshold(clf, X_val, y_val):
    best_t, best_s = 0.5, 0.0
    for t in np.arange(0.30, 0.85, 0.05):
        s = fbeta_score(y_val, predict_thresh(clf, X_val, t), beta=0.5, pos_label='buy', zero_division=0)
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t

# Use the TV set for final threshold tuning
opt_threshold = tune_threshold(best_model, X_tv_scaled, y_tv)

# --- REPORTING ---
def get_metrics(clf, Xs, ys_str, t=0.5):
    yp = predict_thresh(clf, Xs, t)
    return {
        'bal_acc': balanced_accuracy_score(ys_str, yp),
        'f1':      f1_score(ys_str, yp, pos_label='buy', zero_division=0),
        'prec':    precision_score(ys_str, yp, pos_label='buy', zero_division=0),
        'yp':      yp
    }

m_te = get_metrics(best_model, X_test_scaled, y_test, opt_threshold)
m_tr = get_metrics(best_model, X_tv_scaled, y_tv, opt_threshold)

report_lines = [
    "XGBoost Regularized Report", "=" * 55,
    f"Best Params: {search.best_params_}",
    f"Probability Threshold: {opt_threshold:.2f}",
    f"\nMetric                Train      Test",
    "-" * 45,
    f"{'Balanced Acc':<18} {m_tr['bal_acc']:>8.4f} {m_te['bal_acc']:>8.4f}",
    f"{'F1 (buy)':<18} {m_tr['f1']:>8.4f} {m_te['f1']:>8.4f}",
    f"{'Precision (buy)':<18} {m_tr['prec']:>8.4f} {m_te['prec']:>8.4f}",
    "\nClassification Report (Test):",
    classification_report(y_test, m_te['yp'], zero_division=0)
]

with open(REPORT, 'w') as f:
    f.write("\n".join(report_lines))

print(f"Done. Best Bal Acc (CV): {search.best_score_:.4f}")
print(f"Report saved to {REPORT}")