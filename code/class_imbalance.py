import pandas as pd
import os


file_path = os.path.join("data", "cleaned", "pipeline_output.csv")
targets = [
    'stock_t1_price_up', 'stock_t1_volume_up',
    'stock_t2_price_up', 'stock_t2_volume_up',
    'stock_t4_price_up', 'stock_t4_volume_up'
]

def analyze_imbalance(path):
    if not os.path.exists(path):
        print(f"Error: Could not find file at {path}")
        return


    df = pd.read_csv(path)
    total_rows = len(df)
    
    print(f"Analysis for {total_rows} total samples:")
    print("-" * 50)
    print(f"{'Target Variable':<25} | {'Class':<6} | {'Count':<8} | {'Percentage'}")
    print("-" * 50)


    for col in targets:
        if col not in df.columns:
            print(f"Column {col} missing from CSV.")
            continue
            
        counts = df[col].value_counts().to_dict()
        
        for val in [0, 1]:
            count = counts.get(val, 0)
            percentage = (count / total_rows) * 100
            label = "Down/Flat" if val == 0 else "Up"
            
            print(f"{col:<25} | {val:<6} | {count:<8} | {percentage:.2f}%")
        print("-" * 50)

if __name__ == "__main__":
    analyze_imbalance(file_path)