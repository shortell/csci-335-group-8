import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = os.getcwd()
EMBED_PATH = os.path.join(BASE_DIR, "data", "vector_embeddings", "all-MiniLM-L6-v2", "all-MiniLM-L6-v2_pca.npz")
DATA_PATH  = os.path.join(BASE_DIR, "data", "cleaned", "pipeline_output.csv")
REPORT_DIR = os.path.join(BASE_DIR, "models")
REPORT_FILE = os.path.join(REPORT_DIR, "random_forest_report.txt")

os.makedirs(REPORT_DIR, exist_ok=True)

def run_training():
    # Load Data
    with np.load(EMBED_PATH) as data:
        X_embed = data[data.files[0]] 
    
    df = pd.read_csv(DATA_PATH)
    
    # Append mentions_tesla to the PCA vector
    tesla_feature = df[['mentions_tesla']].values
    X = np.hstack((X_embed, tesla_feature))
    
    targets = [
        'stock_t1_price_up', 'stock_t1_volume_up',
        'stock_t2_price_up', 'stock_t2_volume_up',
        'stock_t4_price_up', 'stock_t4_volume_up'
    ]
    
    model_stats = []

    # Train and Evaluate 6 Random Forests
    for target in targets:
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Pragmatic Hyperparameters for Laptop Performance:
        # n_estimators=100: Good balance of accuracy and speed.
        # max_depth=10: Prevents the forest from growing too complex for ~2.7k samples.
        # n_jobs=-1: Uses all available CPU cores to speed up training.
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            n_jobs=-1, 
            random_state=42
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Capture metrics
        acc = accuracy_score(y_test, y_pred)
        model_stats.append({
            'target': target,
            'accuracy': acc,
            'report': classification_report(y_test, y_pred)
        })

    # Rank by Accuracy 
    model_stats.sort(key=lambda x: x['accuracy'], reverse=True)

    # Write Detailed Report
    with open(REPORT_FILE, "w") as f:
        f.write("RANDOM FOREST COMPREHENSIVE REPORT\n")
        f.write("Feature Set: PCA Embeddings + Mentions_Tesla Boolean\n")
        f.write("=" * 60 + "\n\n")
        
        for i, stats in enumerate(model_stats, 1):
            f.write(f"{i}. TARGET: {stats['target']}\n")
            f.write(f"Overall Accuracy: {stats['accuracy']:.4f}\n")
            f.write("-" * 30 + "\n")
            f.write(stats['report'])
            f.write("\n" + "=" * 60 + "\n\n")

    print(f"Random Forest report generated: {REPORT_FILE}")

if __name__ == "__main__":
    run_training()