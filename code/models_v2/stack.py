import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import BaseEstimator, ClassifierMixin

# Define file paths
input_file = r'data\final\musk_events_k10_replies_True.csv'
output_file = r'stack_report.txt'

df = pd.read_csv(input_file)

features = [
    'mentions_tesla', 'is_reply', 'is_quote', 'is_retweet',
    'positive', 'negative', 'neutral', 'close_delta_z',
    'volume_delta_z', 'price_cv', 'volume_cv'
]

X = df[features].copy()
target = 'close_t1_z'

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

thresh_down, thresh_up = -0.2, 0.2

def categorize_z(z):
    if pd.isna(z):
        return np.nan
    if z < thresh_down:
        return 'down'
    elif z > thresh_up:
        return 'up'
    else:
        return 'flat'
        
y = df[target].apply(categorize_z)
valid_idx = y.notna()
X_valid = X_imputed[valid_idx]
y_valid = y[valid_idx]

# Split data
try:
    X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.2, random_state=42, stratify=y_valid)
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Label encoding for XGBoost and Stacking compatibility
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Custom Wrapper for XGBoost to enforce balanced sample weights internally
class BalancedXGBClassifier(XGBClassifier):
    def fit(self, X, y, **kwargs):
        sample_weights = compute_sample_weight(class_weight='balanced', y=y)
        kwargs['sample_weight'] = sample_weights
        return super().fit(X, y, **kwargs)

# Define Base Models
estimators = [
    ('rf', RandomForestClassifier(
        class_weight='balanced', 
        max_depth=10, 
        min_samples_split=2, 
        n_estimators=50, 
        random_state=42
    )),
    ('xgb', BalancedXGBClassifier(
        learning_rate=0.1, 
        max_depth=7, 
        n_estimators=100, 
        subsample=0.8,
        eval_metric='mlogloss',
        random_state=42
    )),
    ('mlp', MLPClassifier(
        activation='relu', 
        alpha=0.001, 
        hidden_layer_sizes=(11, 11, 11, 11), 
        learning_rate_init=0.001,
        max_iter=2000, 
        random_state=42
    )),
    ('svm', SVC(
        C=1.0, 
        class_weight='balanced', 
        gamma='scale', 
        kernel='linear',
        probability=True, 
        random_state=42
    ))
]

# Build Soft Voting Classifier
vote_clf = VotingClassifier(
    estimators=estimators, 
    voting='soft',
    n_jobs=-1
)

print("Training Soft Voting Ensemble...")
vote_clf.fit(X_train_scaled, y_train_encoded)

# Predict
y_pred_encoded = vote_clf.predict(X_test_scaled)
y_pred = le.inverse_transform(y_pred_encoded)

# Evaluate
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
rep = classification_report(y_test, y_pred, zero_division=0)

report_content = "Soft Voting Ensemble Report\n"
report_content += "===========================\n\n"
report_content += "Base Models:\n"
report_content += " - Random Forest (Model 19 params, balanced)\n"
report_content += " - XGBoost (Model 45 params, balanced weights wrapper)\n"
report_content += " - MLP (Model 11 params, 4 deep layers)\n"
report_content += " - SVM (Model 14 params, linear, probability=True)\n\n"
report_content += "Combination Strategy: Soft Voting (Averaged Probabilities)\n"
report_content += f"Thresholds: down < {thresh_down}, up > {thresh_up}\n\n"
report_content += f"Accuracy: {acc:.4f}\n"
report_content += f"Weighted F1-Score: {f1:.4f}\n\n"
report_content += f"Classification Report:\n{rep}\n"

with open(output_file, 'w') as f:
    f.write(report_content)

print(f"\nReport successfully saved to {output_file}")
print(f"Final Architecture Accuracy: {acc:.4f}")
print(f"Final Architecture Weighted F1: {f1:.4f}")
