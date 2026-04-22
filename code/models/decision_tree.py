import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = os.getcwd()
EMBED_PATH = os.path.join(BASE_DIR, "data", "vector_embeddings", "all-MiniLM-L6-v2", "all-MiniLM-L6-v2_pca.npz")
DATA_PATH  = os.path.join(BASE_DIR, "data", "cleaned", "pipeline_output.csv")
REPORT_DIR = os.path.join(BASE_DIR, "models")
REPORT_FILE = os.path.join(REPORT_DIR, "decision_tree_report.txt")

os.makedirs(REPORT_DIR, exist_ok=True)

def run_training():
    # Load PCA Embeddings
    with np.load(EMBED_PATH) as data:
        X_embed = data[data.files[0]] 
    
    df = pd.read_csv(DATA_PATH)
    
    # FEATURE VERIFICATION: Appending mentions_tesla to the vector
    # We take the boolean column and stack it horizontally to the end of the PCA array.
    tesla_feature = df[['mentions_tesla']].values
    X = np.hstack((X_embed, tesla_feature))
    
    targets = [
        'stock_t1_price_up', 'stock_t1_volume_up',
        'stock_t2_price_up', 'stock_t2_volume_up',
        'stock_t4_price_up', 'stock_t4_volume_up'
    ]
    
    model_stats = []

    # Train and Evaluate 6 Trees
    for target in targets:
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = DecisionTreeClassifier(
            max_depth=5, 
            min_samples_leaf=15, 
            random_state=42
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Capture metrics
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        model_stats.append({
            'target': target,
            'accuracy': acc,
            'report': classification_report(y_test, y_pred) # Full string for the file
        })

    # Rank by Accuracy 
    model_stats.sort(key=lambda x: x['accuracy'], reverse=True)

    with open(REPORT_FILE, "w") as f:
        f.write("DECISION TREE COMPREHENSIVE REPORT\n")
        f.write("Feature Set: PCA Embeddings + Mentions_Tesla Boolean\n")
        f.write("=" * 60 + "\n\n")
        
        for i, stats in enumerate(model_stats, 1):
            f.write(f"{i}. TARGET: {stats['target']}\n")
            f.write(f"Overall Accuracy: {stats['accuracy']:.4f}\n")
            f.write("-" * 30 + "\n")
            f.write(stats['report'])
            f.write("\n" + "=" * 60 + "\n\n")

    print(f"Detailed report generated: {REPORT_FILE}")

if __name__ == "__main__":
    run_training()