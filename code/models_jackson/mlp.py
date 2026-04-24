import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, fbeta_score, classification_report
from sklearn.impute import SimpleImputer

INPUT  = r'data\final\musk_events_k10_replies_True.csv'
REPORT = r'mlp_report.txt'
TARGET = 'max_z_next5'
THRESH = 1.5
FEATURES = [
    'mentions_tesla', 'is_reply', 'is_quote', 'is_retweet',
    'positive', 'negative', 'neutral',
    'close_delta_z', 'volume_delta_z',
    'price_cv', 'volume_cv',
]

df = pd.read_csv(INPUT)
X = SimpleImputer(strategy='median').fit_transform(
    df[FEATURES].apply(pd.to_numeric, errors='coerce')
).astype(np.float64)

y_raw = df[TARGET].apply(lambda z: 'buy' if not pd.isna(z) and z > THRESH else ('dont_buy' if not pd.isna(z) else np.nan))
mask = y_raw.notna().values
X, y = X[mask], np.array(y_raw[y_raw.notna()])

X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.2, random_state=42, stratify=y_tv)

scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train).astype(np.float64)
Xva = scaler.transform(X_val).astype(np.float64)
Xte = scaler.transform(X_test).astype(np.float64)

def predict_thresh(clf, Xs, t=0.5):
    buy_idx = list(clf.classes_).index('buy')
    proba = clf.predict_proba(Xs.astype(np.float64))
    return np.where(proba[:, buy_idx] >= t, 'buy', 'dont_buy')

def tune_threshold(clf, Xva, y_val):
    best_t, best_s = 0.5, 0.0
    for t in np.arange(0.30, 0.85, 0.05):
        s = fbeta_score(y_val, predict_thresh(clf, Xva, t), beta=0.5, pos_label='buy', zero_division=0)
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t

def get_metrics(clf, Xs, ys, t=0.5):
    yp = predict_thresh(clf, Xs, t)
    return {
        'bal_acc': balanced_accuracy_score(ys, yp),
        'f1':      f1_score(ys, yp, pos_label='buy', zero_division=0),
        'prec':    precision_score(ys, yp, pos_label='buy', zero_division=0),
    }

# UPDATED PARAMETER GRID
param_grid = {
    'hidden_layer_sizes': [(11,), (11, 5), (11, 5, 2)],
    'activation': ['relu', 'tanh'],
    'learning_rate_init': [0.001, 0.01],
}

results = []
for params in ParameterGrid(param_grid):
    try:
        clf = MLPClassifier(
            solver='adam',
            max_iter=500,
            random_state=42,
            **params # Dynamically passes activation and other params
        )
        clf.fit(Xtr, y_train)
        m = get_metrics(clf, Xva, y_val)
        composite = (m['bal_acc'] + m['f1'] + m['prec']) / 3
        results.append({**m, 'composite': composite, 'params': params, 'clf': clf})
    except Exception as e:
        print(f"  SKIP {params}: {e}")
        continue

if not results:
    print("All combinations failed."); exit()

best_acc       = max(results, key=lambda r: r['bal_acc'])
best_f1        = max(results, key=lambda r: r['f1'])
best_composite = max(results, key=lambda r: r['composite'])
best_composite['threshold'] = tune_threshold(best_composite['clf'], Xva, y_val)

n, b = len(y), (y == 'buy').sum()
lines = [
    "MLP Report", "=" * 55,
    f"Target: {TARGET}  |  Threshold > {THRESH}",
    f"Total: {n}  |  Buy: {b} ({100*b/n:.1f}%)  |  No-Buy: {n-b} ({100*(n-b)/n:.1f}%)",
    f"Train: {len(y_train)}  |  Val: {len(y_val)}  |  Test: {len(y_test)}",
    f"Features ({len(FEATURES)}): {', '.join(FEATURES)}", "",
]

def model_block(title, r, t=0.5):
    m_tr = get_metrics(r['clf'], Xtr, y_train, t)
    m_va = get_metrics(r['clf'], Xva, y_val, t)
    m_te = get_metrics(r['clf'], Xte, y_test, t)
    note = f"  (prob threshold: {t:.2f})" if t != 0.5 else ""
    block = [f"[ {title} ]", f"Params: {r['params']}{note}",
             f"{'Metric':<18} {'Train':>8} {'Val':>8} {'Test':>8}", "-" * 45]
    for k, label in [('bal_acc', 'Balanced Acc'), ('f1', 'F1 (buy)'), ('prec', 'Precision (buy)')]:
        block.append(f"{label:<18} {m_tr[k]:>8.4f} {m_va[k]:>8.4f} {m_te[k]:>8.4f}")
    block.append("\nClassification Report (Test):")
    block.append(classification_report(y_test, predict_thresh(r['clf'], Xte, t), zero_division=0))
    return "\n".join(block)

lines.append(model_block("Best Balanced Accuracy",       best_acc))
lines.append(model_block("Best F1 (buy)",                 best_f1))
lines.append(model_block("Best Composite (Bal+F1+Prec)", best_composite, best_composite['threshold']))

with open(REPORT, 'w') as f:
    f.write("\n".join(lines))
print(f"Done. Report: {REPORT}")