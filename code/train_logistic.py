import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

"""
train_logistic.py

Trains a Logistic Regression classifier to predict TSLA price movement
direction from Elon Musk tweet embeddings.

Inputs
  data/vector_embeddings/<provider>/<model>.npz  — tweet embeddings
  data/cleaned/pipeline_output.csv               — tweet metadata + aligned stock prices

Output
  Prints classification report and confusion matrix.
  Saves plots to analysis/logistic_regression_results.png
"""

# ── CONFIG ────────────────────────────────────────────────────────────────────

ACTIVE_MODEL = ('open_ai', 'text-embedding-3-small')

BASE_DIR       = Path(__file__).parent.parent
EMBEDDING_PATH = BASE_DIR / 'data' / 'vector_embeddings' / ACTIVE_MODEL[0] / f'{ACTIVE_MODEL[1]}.npz'
PIPELINE_PATH  = BASE_DIR / 'data' / 'cleaned' / 'pipeline_output.csv'
PLOT_OUT       = BASE_DIR / 'analysis' / 'logistic_regression_results.png'

BASELINE_COL   = 'stock_t0_close'
LOOKAHEAD_COL  = 'stock_t2_close' # can change to t1, t4, t8, t16 for different lookahead windows
PCA_COMPONENTS = 400


def load_embeddings_with_features():
    """
    Creates the feature space with tweet embeddings + engagement features.
    Feature scaling the engagement features.
    """
    data       = np.load(EMBEDDING_PATH, allow_pickle=False)
    X_emb      = data['embeddings']
    row_ids    = data['row_ids']
    print(f"Embeddings loaded: {X_emb.shape}")

    df         = pd.read_csv(PIPELINE_PATH)
    # log1p compresses skewed engagement counts (viral tweets can have 500k likes 
    # vs most having <100) so outliers don't dominate the feature space or PCA
    engagement = np.log1p(df[['likeCount', 'retweetCount', 'replyCount']].fillna(0).to_numpy())

    X = np.hstack([X_emb, engagement]) # stacking engagement features with embeddings for model input
    print(f"Feature matrix with engagement: {X.shape}")
    return X, row_ids


def build_labels(row_ids: np.ndarray) -> np.ndarray:
    """
    Build binary labels based on whether the stock price went up (1) or down (0) 
    compared to the baseline. Handles missing stock data by assigning NaN labels.
    """
    df      = pd.read_csv(PIPELINE_PATH)
    labels  = (df[LOOKAHEAD_COL] > df[BASELINE_COL]).astype(float)
    missing = df[LOOKAHEAD_COL].isna() | df[BASELINE_COL].isna()
    labels[missing] = np.nan
    return labels.to_numpy()


def align_and_clean(X, labels):
    """
    Aligns the feature matrix with the labels using the row_ids and drops samples with missing labels.
    Also prints class balance and how many samples were dropped due to missing stock data.
    """
    mask    = ~np.isnan(labels)
    X_clean = X[mask]
    y_clean = labels[mask].astype(int)
    print(f"Dropped {(~mask).sum()} tweets with no matching stock data")
    print(f"Remaining: {len(y_clean)} samples | class balance: {y_clean.mean():.2%} positive")
    return X_clean, y_clean


def apply_pca(X_train, X_test, n_components):
    """
    Applying the PCA dimensionality reduction technique to the embedding features 
    while keeping the engagement features intact.This helps to reduce the dimensionality
    of the embedding space while preserving the variance, and prevents the engagement 
    features from dominating the PCA components.

    notes: while training I increased the PCA components to 400 from 100 to retain more variance and improve performance.
    """
    if n_components is None:
        return X_train, X_test, None

    # PCA on embeddings only — keeps engagement features from dominating components
    emb_train, eng_train = X_train[:, :-3], X_train[:, -3:]
    emb_test,  eng_test  = X_test[:, :-3],  X_test[:, -3:]

    pca         = PCA(n_components=n_components, random_state=42)
    emb_train_r = pca.fit_transform(emb_train)
    emb_test_r  = pca.transform(emb_test)

    print(f"PCA: {emb_train.shape[1]}d -> {n_components}d | "
          f"variance retained: {pca.explained_variance_ratio_.sum():.2%}")
    return np.hstack([emb_train_r, eng_train]), np.hstack([emb_test_r, eng_test]), pca


def train(X_train, y_train):
    """
    Trains a Logistic Regression classifier with balanced class weights to handle class imbalance.
    Also applies feature scaling to the training data using StandardScaler.
    """
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model   = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model, scaler


def evaluate(model, scaler, X_test, y_test):
    """
    Evaluates the trained model on the test set and prints classification metrics.
    """
    y_pred = model.predict(scaler.transform(X_test))
    print("\n── Classification Report ─────────────────────────")
    print(classification_report(y_test, y_pred, target_names=['Down (0)', 'Up (1)']))
    print("── Confusion Matrix ──────────────────────────────")
    print(confusion_matrix(y_test, y_pred))
    return y_pred


def plot_results(pca, y, y_test, y_pred):
    """
    Generates a 3-panel plot showing:
        1. PCA explained variance curve to justify the choice of components.
        2. Class balance bar chart to visualize the distribution of up/down samples.
        3. Confusion matrix heatmap to show true vs predicted labels and overall accuracy.
    """
    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(
        f"Logistic Regression — TSLA Price Direction from Tweet Embeddings\n"
        f"Model: {ACTIVE_MODEL[1]}  |  Lookahead: {LOOKAHEAD_COL}",
        fontsize=11, y=1.02
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # PCA explained variance
    ax1 = fig.add_subplot(gs[0])
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax1.plot(range(1, len(cumvar) + 1), cumvar, color='steelblue', linewidth=1.5)
    ax1.axhline(cumvar[PCA_COMPONENTS - 1], color='tomato', linestyle='--', linewidth=1,
                label=f'{PCA_COMPONENTS} components = {cumvar[PCA_COMPONENTS - 1]:.1f}%')
    ax1.axvline(PCA_COMPONENTS, color='tomato', linestyle='--', linewidth=1)
    ax1.set_xlabel('Number of PCA Components')
    ax1.set_ylabel('Cumulative Variance Retained (%)')
    ax1.set_title('PCA Explained Variance')
    ax1.legend(fontsize=8)
    ax1.set_xlim(1, len(cumvar))
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)

    # Class balance
    ax2    = fig.add_subplot(gs[1])
    n_up   = int(y.sum())
    n_down = len(y) - n_up
    bars   = ax2.bar(['Down (0)', 'Up (1)'], [n_down, n_up],
                     color=['tomato', 'steelblue'], width=0.5, edgecolor='white')
    for bar, count in zip(bars, [n_down, n_up]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f'{count}\n({count/len(y)*100:.1f}%)',
            ha='center', va='bottom', fontsize=9
        )
    ax2.set_title('Class Balance')
    ax2.set_ylabel('Number of Tweets')
    ax2.set_ylim(0, max(n_up, n_down) * 1.2)
    ax2.grid(True, alpha=0.3, axis='y')

    # Confusion matrix heatmap
    ax3 = fig.add_subplot(gs[2])
    cm  = confusion_matrix(y_test, y_pred)
    im  = ax3.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax3.text(j, i, str(cm[i, j]),
                     ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black',
                     fontsize=13, fontweight='bold')
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Pred Down', 'Pred Up'])
    ax3.set_yticklabels(['Actual Down', 'Actual Up'])
    ax3.set_title(f'Confusion Matrix\nAccuracy: {(cm.diagonal().sum()/cm.sum())*100:.1f}%')
    ax3.set_xlabel('Predicted Label')
    ax3.set_ylabel('True Label')

    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_OUT, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {PLOT_OUT}")
    plt.show()


if __name__ == '__main__':
    X, row_ids = load_embeddings_with_features()

    labels = build_labels(row_ids)
    X, y   = align_and_clean(X, labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    X_train, X_test, pca = apply_pca(X_train, X_test, PCA_COMPONENTS)
    model, scaler         = train(X_train, y_train)
    y_pred                = evaluate(model, scaler, X_test, y_test)

    plot_results(pca, y, y_test, y_pred)