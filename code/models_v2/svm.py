import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.impute import SimpleImputer
import os

# Define file paths
input_file = r'data\final\musk_events_k10_replies_True.csv'
output_file = r'svm_report.txt'

# Load the data
df = pd.read_csv(input_file)

# Features
features = [
    'mentions_tesla', 'is_reply', 'is_quote', 'is_retweet',
    'positive', 'negative', 'neutral', 'close_delta_z',
    'volume_delta_z', 'price_cv', 'volume_cv'
]

missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print(f"Missing features: {missing_features}")

X = df[features].copy()
target = 'close_t1_z'

report_content = "SVM Models Report (Tuning)\n"
report_content += "==========================\n\n"

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

thresholds_list = [
    (-0.2, 0.2)
]

param_grid = {
    'C': [0.1, 1.0, 10.0, 100.0],
    'gamma': ['scale', 0.01],
    'kernel': ['rbf', 'linear'],
    'class_weight': [None, 'balanced']
}

sampler = list(ParameterGrid(param_grid))

best_acc = -1
best_model_info = {}

for i in range(len(sampler)):
    thresh_down, thresh_up = thresholds_list[0]
    params = sampler[i]
    
    model_report = ""
    model_report += f"Model {i+1}:\n"
    model_report += f"Thresholds: down < {thresh_down}, up > {thresh_up}\n"
    model_report += f"Params: {params}\n"
    
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
    
    if len(y_valid) == 0:
        continue
        
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.2, random_state=42, stratify=y_valid)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.2, random_state=42)
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = SVC(random_state=42, **params)
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    rep = classification_report(y_test, y_pred, zero_division=0)
    
    if acc > 0.6:
        model_report += f"Accuracy: {acc:.4f}\n"
        model_report += f"Weighted F1-Score: {f1:.4f}\n\n"
        report_content += model_report
    
    if acc > best_acc:
        best_acc = acc
        best_model_info = {
            'thresholds': (thresh_down, thresh_up),
            'params': params,
            'f1': f1,
            'report': rep
        }

report_content += "=================================\n"
report_content += "BEST PERFORMING MODEL\n"
report_content += "=================================\n"
report_content += f"Accuracy: {best_acc:.4f}\n"
report_content += f"Weighted F1-Score: {best_model_info.get('f1'):.4f}\n"
report_content += f"Thresholds: {best_model_info.get('thresholds')}\n"
report_content += f"Params: {best_model_info.get('params')}\n\n"
report_content += f"Classification Report:\n{best_model_info.get('report')}\n"

with open(output_file, 'w') as f:
    f.write(report_content)

print(f"Report saved to {output_file}")
print(f"Best accuracy: {best_acc:.4f}")
