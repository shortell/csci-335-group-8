import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # Path to the data file
    file_path = os.path.join("data", "final", "musk_events_k10_replies_True.csv")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Load the data
    df = pd.read_csv(file_path)
    
    # Verify the required sentiment columns exist
    sentiment_cols = ['positive', 'negative', 'neutral']
    if not all(col in df.columns for col in sentiment_cols):
        print(f"Error: Missing one of the required sentiment columns in {file_path}")
        return

    # Determine dominant sentiment class by finding the column with the max value
    df['dominant_sentiment'] = df[sentiment_cols].idxmax(axis=1)

    # Calculate distribution
    distribution = df['dominant_sentiment'].value_counts()
    
    # Print the absolute and percentage distributions
    print(f"--- Class Distribution ({len(df)} total tweets) ---")
    print(distribution)
    print("\n--- Percentage Distribution ---")
    print((distribution / len(df) * 100).round(2).astype(str) + '%')

    # Plot the distribution
    plt.figure(figsize=(8, 6))
    
    # Create the bar plot
    ax = sns.barplot(x=distribution.index, y=distribution.values, palette='viridis')
    plt.title('Tweet Sentiment Class Distribution', fontsize=15)
    plt.xlabel('Sentiment Class', fontsize=12)
    plt.ylabel('Number of Tweets', fontsize=12)
    
    # Annotate the counts above the bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', 
                    fontsize=12, color='black', xytext=(0, 5), 
                    textcoords='offset points')

    # Beautify
    sns.despine()
    plt.tight_layout()
    
    # Save and show the plot
    output_img = 'sentiment_distribution.png'
    plt.savefig(output_img)
    print(f"\nPlot successfully saved as '{output_img}'.")
    plt.show()

if __name__ == "__main__":
    main()
