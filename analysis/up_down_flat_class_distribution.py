import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your current final dataset
file_path = r'data\final\musk_events_k10_replies_True.csv'
df = pd.read_csv(file_path)

# 2. Define the 10 target variables
targets = [f'close_t{i}_z' for i in range(1, 6)] + [f'volume_t{i}_z' for i in range(1, 6)]

def categorize_z(z):
    """Categorizes Z-scores into Up, Down, or Flat based on 0.5 threshold."""
    if z > 0.5:
        return 'Up'
    elif z < -0.5:
        return 'Down'
    else:
        return 'Flat'

# 3. Print Results and Plot
print(f"{'Target Variable':<15} | {'Down':<8} | {'Flat':<8} | {'Up':<8} | {'Total'}")
print("-" * 65)

for target in targets:
    # Categorize temporarily for this display
    labels = df[target].apply(categorize_z)
    
    # Calculate distribution
    counts = labels.value_counts().reindex(['Down', 'Flat', 'Up'], fill_value=0)
    
    # Print numerical distribution to terminal
    print(f"{target:<15} | {counts['Down']:<8} | {counts['Flat']:<8} | {counts['Up']:<8} | {len(df)}")
    
    # 4. Generate Plot (Interactive Window)
    plt.figure(figsize=(8, 5))
    counts.plot(kind='bar', color=['#e74c3c', '#95a5a6', '#2ecc71']) # Red, Gray, Green
    plt.title(f'Class Distribution: {target} (Threshold ±0.5)')
    plt.ylabel('Frequency')
    plt.xlabel('Market Reaction Category')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Opens the plot in a window (Script pauses until you close the window)
    plt.show()

print("\n--- Summary Complete: No files were modified or saved. ---")