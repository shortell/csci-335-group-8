import os
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- 1. MODEL SETUP ---
MODEL_PATH = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

def compute_df_sentiment_batched(df, batch_size=16):
    """
    Core logic: Processes tweets in chunks and aligns them with original rows.
    """
    print(f"Analyzing sentiment for {len(df)} rows in batches of {batch_size}...")
    
    texts = df['whole_text'].fillna("").tolist()
    all_probs = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = [str(t).replace("@", "@user").replace("http", "http") 
                       for t in texts[i:i+batch_size]]
        
        # Tokenize with padding and truncation (fixed the 512 error)
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Softmax to get probabilities 0.0 to 1.0
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()
            all_probs.extend(probs)

    # Align results back to the original dataframe index
    probs_df = pd.DataFrame(all_probs, columns=['negative', 'neutral', 'positive'], index=df.index)
    df[['negative', 'neutral', 'positive']] = probs_df.round(4)
    
    return df

def run_sentiment_pipeline(input_file_path: str):
    """
    The wrapper: Handles reading/writing so you don't recompute.
    """
    input_path = Path(input_file_path)
    if not input_path.exists():
        print(f"Error: File not found at {input_file_path}")
        return

    # 1. Load the cleaned data from your previous pipeline
    df = pd.read_csv(input_path)
    
    # 2. Run the optimized batch processing
    df = compute_df_sentiment_batched(df)

    # 3. Create 'data/final' and save so you never have to run this again
    base_dir = input_path.parent.parent
    final_dir = base_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = final_dir / input_path.name
    df.to_csv(output_path, index=False)
    
    print(f"Successfully saved final analysis to: {output_path}")
    print(f"Total rows processed: {len(df)}")

if __name__ == "__main__":
    # Point this to your cleaned CSV
    target_file = "data/cleaned/musk_events_k10_replies_False.csv"
    run_sentiment_pipeline(target_file)