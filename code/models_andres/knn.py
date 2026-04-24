import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

from loader import load_data

# LOAD
data = load_data()
X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
y_train_cls, y_val_cls, y_test_cls = data["y_train_cls"], data["y_val_cls"], data["y_test_cls"]
y_train_reg, y_val_reg, y_test_reg = data["y_train_reg"], data["y_val_reg"], data["y_test_reg"]

CLASSES = ["down", "neutral", "up"]

# CLASSIFICATION
cls_model = KNeighborsClassifier(n_neighbors=5)
cls_model.fit(X_train, y_train_cls)

print("=" * 50)
print("KNN — Classification (Val)")
print("=" * 50)
val_preds = cls_model.predict(X_val)
print(classification_report(y_val_cls, val_preds))

print("=" * 50)
print("KNN — Classification (Test)")
print("=" * 50)
test_preds = cls_model.predict(X_test)
print(classification_report(y_test_cls, test_preds))

# REGRESSION
reg_model = KNeighborsRegressor(n_neighbors=5)
reg_model.fit(X_train, y_train_reg)

print("=" * 50)
print("KNN — Regression (Val)")
print("=" * 50)
val_reg_preds = reg_model.predict(X_val)
print(f"  RMSE: {np.sqrt(mean_squared_error(y_val_reg, val_reg_preds)):.4f}")
print(f"  MAE:  {mean_absolute_error(y_val_reg, val_reg_preds):.4f}")
print(f"  R(squared):   {r2_score(y_val_reg, val_reg_preds):.4f}")

print("\n" + "=" * 50)
print("KNN — Regression (Test)")
print("=" * 50)
test_reg_preds = reg_model.predict(X_test)
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_reg, test_reg_preds)):.4f}")
print(f"  MAE:  {mean_absolute_error(y_test_reg, test_reg_preds):.4f}")
print(f"  R(squared):   {r2_score(y_test_reg, test_reg_preds):.4f}")

# CLASSIFICATION PLOTS
val_report  = classification_report(y_val_cls,  val_preds,  output_dict=True)
test_report = classification_report(y_test_cls, test_preds, output_dict=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("KNN — Classification Results", fontsize=15, fontweight="bold")

# F1 per class — Val vs Test
ax = axes[0]
f1_val  = [val_report[c]["f1-score"]  for c in CLASSES]
f1_test = [test_report[c]["f1-score"] for c in CLASSES]
x, w = np.arange(len(CLASSES)), 0.35
b1 = ax.bar(x - w/2, f1_val,  w, label="Val",  color="#3498db", alpha=0.85)
b2 = ax.bar(x + w/2, f1_test, w, label="Test", color="#9b59b6", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(CLASSES)
ax.set_ylim(0, 1); ax.set_ylabel("F1 Score")
ax.set_title("F1 Score per Class"); ax.legend()
ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=8)
ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=8)

# Precision / Recall / F1 grouped — Test only
ax = axes[1]
metrics = ["precision", "recall", "f1-score"]
colors  = ["#e74c3c", "#3498db", "#2ecc71"]
for i, (metric, color) in enumerate(zip(metrics, colors)):
    vals = [test_report[c][metric] for c in CLASSES]
    bars = ax.bar(x + (i - 1) * 0.25, vals, 0.25,
                  label=metric.replace("-", " ").title(), color=color, alpha=0.85)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(CLASSES)
ax.set_ylim(0, 1); ax.set_ylabel("Score")
ax.set_title("Precision / Recall / F1 (Test)"); ax.legend(fontsize=8)

# Confusion matrix — Test
ax = axes[2]
cm = confusion_matrix(y_test_cls, test_preds, labels=CLASSES)
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix (Test)")
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("knn_classification.png", dpi=150)
plt.show()
print("Saved: knn_classification.png")

# REGRESSION PLOTS
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("KNN — Regression Results", fontsize=15, fontweight="bold")

# Metrics bar chart — Val vs Test
ax = axes[0]
metric_labels = ["RMSE", "MAE", "R(squared)"]
val_scores  = [np.sqrt(mean_squared_error(y_val_reg, val_reg_preds)),
               mean_absolute_error(y_val_reg, val_reg_preds),
               r2_score(y_val_reg, val_reg_preds)]
test_scores = [np.sqrt(mean_squared_error(y_test_reg, test_reg_preds)),
               mean_absolute_error(y_test_reg, test_reg_preds),
               r2_score(y_test_reg, test_reg_preds)]
x = np.arange(len(metric_labels))
b1 = ax.bar(x - w/2, val_scores,  w, label="Val",  color="#3498db", alpha=0.85)
b2 = ax.bar(x + w/2, test_scores, w, label="Test", color="#9b59b6", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(metric_labels)
ax.set_title("Regression Metrics (Val vs Test)"); ax.set_ylabel("Score"); ax.legend()
ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=8)
ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=8)

# Actual vs Predicted scatter — Test
ax = axes[1]
ax.scatter(y_test_reg, test_reg_preds, alpha=0.4, s=15, color="#3498db")
lims = [min(y_test_reg.min(), test_reg_preds.min()),
        max(y_test_reg.max(), test_reg_preds.max())]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
ax.set_xlabel("Actual max_z_next5"); ax.set_ylabel("Predicted")
ax.set_title("Actual vs Predicted (Test)"); ax.legend()

# Residual plot — Test
ax = axes[2]
residuals = y_test_reg - test_reg_preds
ax.scatter(test_reg_preds, residuals, alpha=0.4, s=15, color="#e74c3c")
ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
ax.set_xlabel("Predicted"); ax.set_ylabel("Residual (Actual − Predicted)")
ax.set_title("Residual Plot (Test)")

plt.tight_layout()
plt.savefig("knn_regression.png", dpi=150)
plt.show()
print("Saved: knn_regression.png")