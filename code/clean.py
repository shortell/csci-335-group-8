# import pandas as pd
# import re
# import html
# import numpy as np
# import os

# # --- 1. GLOBAL PATHS ---
# # Update these to the actual locations of your files
# POSTS_PATH  = "data/all_musk_posts.csv"
# QUOTES_PATH = "data/musk_quote_tweets.csv"
# STOCK_PATH  = "data/TSLA_1min_market_hours_UTC.csv"

# # --- 2. REGEX & SETTINGS ---
# EPS = 1e-8
# URL_RE = re.compile(r"http\S+")
# AT_RE  = re.compile(r"@\w+")
# WS_RE  = re.compile(r"\s+")
# RT_RE  = re.compile(r"^RT\s+")


# TSLA_RE = re.compile(r"\b(?:tesla|tsla|model [3ysx]|cybertruck)\b")

# # --- 3. CLEANING HELPER ---
# def clean_text(text):
#     if pd.isna(text): return ""
#     text = re.sub(URL_RE, "", str(text))
#     text = re.sub(AT_RE, "", text)
#     text = re.sub(RT_RE, "", text)
#     text = html.unescape(text) 
#     return re.sub(WS_RE, " ", text).strip()

# # --- 4. THE PIPELINE ---
# # def run_pipeline(k=10, include_replies=True, save_csv=True):
# #     # Load Data
# #     posts = pd.read_csv(POSTS_PATH, low_memory=False)
# #     quotes = pd.read_csv(QUOTES_PATH)
# #     stock = pd.read_csv(STOCK_PATH)

# #     # Convert Timestamps (Explicitly UTC)
# #     posts["createdAt"] = pd.to_datetime(posts["createdAt"], utc=True)
# #     quotes["musk_quote_created_at"] = pd.to_datetime(quotes["musk_quote_created_at"], utc=True)
# #     stock["timestamp"] = pd.to_datetime(stock["timestamp"], utc=True).sort_values()

# #     # A. Filter Replies
# #     if not include_replies:
# #         posts = posts[posts["isReply"] != True]

# #     # B. Market Hours Filter (09:35 - 15:55 New York Time)
# #     # Using DatetimeIndex to handle time-based slicing
# #     ny_idx = pd.DatetimeIndex(posts["createdAt"]).tz_convert("America/New_York")
# #     posts = posts.iloc[ny_idx.indexer_between_time("09:35", "15:55")]

# #     # C. Build whole_text (Contextual Concatenation)
# #     quotes["clean_q"] = (quotes["orig_tweet_text"].apply(clean_text) + " " + 
# #                          quotes["musk_quote_tweet"].apply(clean_text)).str.strip()
# #     q_map = quotes.set_index("musk_tweet_id")["clean_q"]

# #     posts["whole_text"] = posts["fullText"].apply(clean_text)
    
# #     is_q = posts["isQuote"] == True
# #     posts.loc[is_q, "whole_text"] = posts.loc[is_q, "id"].map(q_map).fillna(posts.loc[is_q, "whole_text"])

# #     if "inReplyToText" in posts.columns:
# #         mask = posts["isReply"] == True
# #         posts.loc[mask, "whole_text"] = (posts.loc[mask, "inReplyToText"].apply(clean_text) + 
# #                                          " " + posts.loc[mask, "whole_text"]).str.strip()

# #     # D. Filtering: Word Count & Isolation (5 min)
# #     posts = posts[posts["whole_text"].str.split().str.len() >= k].copy()
# #     posts = posts.sort_values("createdAt")
# #     diff_prev = posts["createdAt"].diff().dt.total_seconds() / 60
# #     diff_next = posts["createdAt"].diff(-1).dt.total_seconds() / -60
# #     posts = posts[(diff_prev.fillna(999) > 10) & (diff_next.fillna(999) > 5)]

# #     # E. Feature Engineering (Z-Scores & CV)
# #     tweet_ts = posts["createdAt"].dt.tz_localize(None)
# #     stock_ts = stock["timestamp"].dt.tz_localize(None)
# #     stock_data = stock.reset_index(drop=True)
    
# #     final_rows = []
# #     for t_time in tweet_ts:
# #         t_min = t_time.floor("min")
# #         idx = stock_ts.searchsorted(t_min)
        
# #         if 10 <= idx < (len(stock_data) - 5) and stock_ts[idx] == t_min:
# #             pre = stock_data.iloc[idx-10 : idx]
# #             post = stock_data.iloc[idx+1 : idx+6]

# #             m_p, s_p = pre["close"].mean(), pre["close"].std()
# #             m_v, s_v = pre["volume"].mean(), pre["volume"].std()

# #             # --- Existing features ---
# #             close_range = pre["close"].max() - pre["close"].min()

# #             row = {
# #                 "close_delta_z": (pre["close"].iloc[-1] - pre["close"].iloc[0]) / (s_p + EPS),
# #                 "volume_delta_z": (pre["volume"].iloc[-1] - pre["volume"].iloc[0]) / (s_v + EPS),
# #                 "price_cv": s_p / (m_p + EPS),
# #                 "volume_cv": s_v / (m_v + EPS),

# #                 # --- New features ---
# #                 # Where the last bar's close sits within the 10-bar high/low range
# #                 "close_position": (pre["close"].iloc[-1] - pre["close"].min()) / (close_range + EPS),

# #                 # --- Target: highest z-score of close in the next 5 minutes ---
# #                 "max_z_next5": ((post["close"] - m_p) / (s_p + EPS)).max(),
# #             }
# #             final_rows.append(row)
# #         else:
# #             final_rows.append({k: np.nan for k in ["close_delta_z"]})

# #     # F. Metadata & Float Sentiment Initialization
# #     features_df = pd.DataFrame(final_rows, index=posts.index)
# #     meta_df = pd.DataFrame({
# #         "whole_text": posts["whole_text"],
# #         "tweet_time": posts["createdAt"].dt.floor("min").dt.tz_localize(None),  # UTC minute of tweet, for price lookup
# #         "mentions_tesla": posts["whole_text"].str.contains(TSLA_RE).astype(int),
# #         "is_reply": posts["isReply"].fillna(False).astype(int),
# #         "is_quote": posts["isQuote"].fillna(False).astype(int),
# #         "is_retweet": posts["isRetweet"].fillna(False).astype(int),
# #         # "hour": posts["createdAt"].dt.hour,
# #         # "day_of_week": posts["createdAt"].dt.dayofweek,
# #         "positive": 0.0, 
# #         "negative": 0.0, 
# #         "neutral": 0.0
# #     })

# #     # Merge, round to 3 decimal places, and remove incomplete windows
# #     result = pd.concat([meta_df, features_df], axis=1).dropna(subset=["close_delta_z", "max_z_next5"])
# #     result = result.round(3)

# #     if save_csv:
# #         os.makedirs("data/cleaned", exist_ok=True)
# #         filename = f"data/cleaned/musk_events_k{k}_replies_{include_replies}.csv"
# #         result.to_csv(filename, index=False)
# #         print(f"File saved: {filename}")
    
# #     print(f"Total tweets saved: {len(result)}")

# #     return result

# def run_pipeline(k=10, include_replies=True, save_csv=True):
#     # --- Load Data ---
#     posts = pd.read_csv(POSTS_PATH, low_memory=False)
#     quotes = pd.read_csv(QUOTES_PATH)
#     stock = pd.read_csv(STOCK_PATH)

#     # --- 1. INITIAL METRICS ---
#     # Convert timestamps to get trading day counts
#     posts["createdAt"] = pd.to_datetime(posts["createdAt"], utc=True)
#     stock["timestamp"] = pd.to_datetime(stock["timestamp"], utc=True)

#     start_posts = len(posts)
#     start_quotes = len(quotes)
#     start_total_items = start_posts + start_quotes
#     start_trading_days = stock["timestamp"].dt.date.nunique()

#     print(f"--- STARTING STATS ---")
#     print(f"Initial Posts:  {start_posts}")
#     print(f"Initial Quotes: {start_quotes}")
#     print(f"Total Combined: {start_total_items}")
#     print(f"Initial Trading Days in Stock Data: {start_trading_days}")
#     print("-" * 25)

#     # Convert quotes timestamp
#     quotes["musk_quote_created_at"] = pd.to_datetime(quotes["musk_quote_created_at"], utc=True)
#     stock = stock.sort_values("timestamp")

#     # A. Filter Replies
#     if not include_replies:
#         posts = posts[posts["isReply"] != True]

#     # B. Market Hours Filter (09:35 - 15:55 New York Time)
#     ny_idx = pd.DatetimeIndex(posts["createdAt"]).tz_convert("America/New_York")
#     posts = posts.iloc[ny_idx.indexer_between_time("09:35", "15:55")]

#     # C. Build whole_text (Contextual Concatenation)
#     quotes["clean_q"] = (quotes["orig_tweet_text"].apply(clean_text) + " " + 
#                          quotes["musk_quote_tweet"].apply(clean_text)).str.strip()
#     q_map = quotes.set_index("musk_tweet_id")["clean_q"]

#     posts["whole_text"] = posts["fullText"].apply(clean_text)
    
#     is_q = posts["isQuote"] == True
#     posts.loc[is_q, "whole_text"] = posts.loc[is_q, "id"].map(q_map).fillna(posts.loc[is_q, "whole_text"])

#     if "inReplyToText" in posts.columns:
#         mask = posts["isReply"] == True
#         posts.loc[mask, "whole_text"] = (posts.loc[mask, "inReplyToText"].apply(clean_text) + 
#                                          " " + posts.loc[mask, "whole_text"]).str.strip()

#     # D. Filtering: Word Count & Isolation
#     posts = posts[posts["whole_text"].str.split().str.len() >= k].copy()
#     posts = posts.sort_values("createdAt")
#     diff_prev = posts["createdAt"].diff().dt.total_seconds() / 60
#     diff_next = posts["createdAt"].diff(-1).dt.total_seconds() / -60
#     posts = posts[(diff_prev.fillna(999) > 10) & (diff_next.fillna(999) > 5)]

#     # E. Feature Engineering (Z-Scores & CV)
#     tweet_ts = posts["createdAt"].dt.tz_localize(None)
#     stock_ts = stock["timestamp"].dt.tz_localize(None)
#     stock_data = stock.reset_index(drop=True)
    
#     final_rows = []
#     for t_time in tweet_ts:
#         t_min = t_time.floor("min")
#         idx = stock_ts.searchsorted(t_min)
        
#         if 10 <= idx < (len(stock_data) - 5) and stock_ts[idx] == t_min:
#             pre = stock_data.iloc[idx-10 : idx]
#             post = stock_data.iloc[idx+1 : idx+6]
#             m_p, s_p = pre["close"].mean(), pre["close"].std()
#             m_v, s_v = pre["volume"].mean(), pre["volume"].std()
#             close_range = pre["close"].max() - pre["close"].min()

#             row = {
#                 "close_delta_z": (pre["close"].iloc[-1] - pre["close"].iloc[0]) / (s_p + EPS),
#                 "volume_delta_z": (pre["volume"].iloc[-1] - pre["volume"].iloc[0]) / (s_v + EPS),
#                 "price_cv": s_p / (m_p + EPS),
#                 "volume_cv": s_v / (m_v + EPS),
#                 "close_position": (pre["close"].iloc[-1] - pre["close"].min()) / (close_range + EPS),
#                 "max_z_next5": ((post["close"] - m_p) / (s_p + EPS)).max(),
#             }
#             final_rows.append(row)
#         else:
#             final_rows.append({k: np.nan for k in ["close_delta_z"]})

#     # F. Metadata & Final DataFrame assembly
#     features_df = pd.DataFrame(final_rows, index=posts.index)
#     meta_df = pd.DataFrame({
#         "whole_text": posts["whole_text"],
#         "tweet_time": posts["createdAt"].dt.floor("min").dt.tz_localize(None),
#         "mentions_tesla": posts["whole_text"].str.contains(TSLA_RE).astype(int),
#         "is_reply": posts["isReply"].fillna(False).astype(int),
#         "is_quote": posts["isQuote"].fillna(False).astype(int),
#         "is_retweet": posts["isRetweet"].fillna(False).astype(int),
#         "positive": 0.0, "negative": 0.0, "neutral": 0.0
#     })

#     result = pd.concat([meta_df, features_df], axis=1).dropna(subset=["close_delta_z", "max_z_next5"])
#     result = result.round(3)

#     # --- 2. FINAL METRICS ---
#     end_total_items = len(result)
#     end_trading_days = result["tweet_time"].dt.date.nunique()

#     print(f"--- FINAL CLEANED STATS ---")
#     print(f"Final Usable Rows: {end_total_items}")
#     print(f"Final Trading Days: {end_trading_days}")
#     print(f"Net Retention: {(end_total_items / start_total_items) * 100:.2f}%")
#     print("-" * 25)

#     if save_csv:
#         os.makedirs("data/cleaned", exist_ok=True)
#         filename = f"data/cleaned/musk_events_k{k}_replies_{include_replies}.csv"
#         result.to_csv(filename, index=False)
#         print(f"File saved: {filename}")
    
#     return result

# # --- 5. EXECUTION ---
# if __name__ == "__main__":
#     final_df = run_pipeline(k=10, include_replies=False, save_csv=False)
import pandas as pd
import re
import html
import numpy as np
import os

# --- 1. GLOBAL PATHS ---
POSTS_PATH  = "data/all_musk_posts.csv"
QUOTES_PATH = "data/musk_quote_tweets.csv"
STOCK_PATH  = "data/TSLA_1min_market_hours_UTC.csv"

# --- 2. REGEX & SETTINGS ---
EPS = 1e-8
URL_RE = re.compile(r"http\S+")
AT_RE  = re.compile(r"@\w+")
WS_RE  = re.compile(r"\s+")
RT_RE  = re.compile(r"^RT\s+")

TSLA_RE = re.compile(r"\b(?:tesla|tsla|model [3ysx]|cybertruck)\b")

# --- 3. CLEANING HELPER ---
def clean_text(text):
    if pd.isna(text): return ""
    text = re.sub(URL_RE, "", str(text))
    text = re.sub(AT_RE, "", text)
    text = re.sub(RT_RE, "", text)
    text = html.unescape(text) 
    return re.sub(WS_RE, " ", text).strip()

# --- 4. THE PIPELINE ---
def run_pipeline(k=10, include_replies=True, save_csv=True):
    # Load Data
    posts = pd.read_csv(POSTS_PATH, low_memory=False)
    quotes = pd.read_csv(QUOTES_PATH)
    stock = pd.read_csv(STOCK_PATH)

    # --- 1. INITIAL METRICS ---
    start_posts = len(posts)
    start_quotes = len(quotes)
    start_combined_sum = start_posts + start_quotes
    # Brief conversion just to count unique dates
    start_trading_days = pd.to_datetime(stock["timestamp"], utc=True).dt.date.nunique()

    print(f"--- STARTING STATS ---")
    print(f"Initial Posts:  {start_posts}")
    print(f"Initial Quotes: {start_quotes}")
    print(f"Total Starting Sum: {start_combined_sum}")
    print(f"Initial Trading Days: {start_trading_days}")
    print("-" * 25)

    # Convert Timestamps (Explicitly UTC) - ORIGINAL LOGIC
    posts["createdAt"] = pd.to_datetime(posts["createdAt"], utc=True)
    quotes["musk_quote_created_at"] = pd.to_datetime(quotes["musk_quote_created_at"], utc=True)
    stock["timestamp"] = pd.to_datetime(stock["timestamp"], utc=True).sort_values()

    # A. Filter Replies
    if not include_replies:
        posts = posts[posts["isReply"] != True]

    # B. Market Hours Filter (09:35 - 15:55 New York Time)
    # Using DatetimeIndex to handle time-based slicing
    ny_idx = pd.DatetimeIndex(posts["createdAt"]).tz_convert("America/New_York")
    posts = posts.iloc[ny_idx.indexer_between_time("09:35", "15:55")]

    # C. Build whole_text (Contextual Concatenation)
    quotes["clean_q"] = (quotes["orig_tweet_text"].apply(clean_text) + " " + 
                         quotes["musk_quote_tweet"].apply(clean_text)).str.strip()
    q_map = quotes.set_index("musk_tweet_id")["clean_q"]

    posts["whole_text"] = posts["fullText"].apply(clean_text)
    
    is_q = posts["isQuote"] == True
    posts.loc[is_q, "whole_text"] = posts.loc[is_q, "id"].map(q_map).fillna(posts.loc[is_q, "whole_text"])

    if "inReplyToText" in posts.columns:
        mask = posts["isReply"] == True
        posts.loc[mask, "whole_text"] = (posts.loc[mask, "inReplyToText"].apply(clean_text) + 
                                         " " + posts.loc[mask, "whole_text"]).str.strip()

    # D. Filtering: Word Count & Isolation (5 min)
    posts = posts[posts["whole_text"].str.split().str.len() >= k].copy()
    posts = posts.sort_values("createdAt")
    diff_prev = posts["createdAt"].diff().dt.total_seconds() / 60
    diff_next = posts["createdAt"].diff(-1).dt.total_seconds() / -60
    posts = posts[(diff_prev.fillna(999) > 10) & (diff_next.fillna(999) > 5)]

    # E. Feature Engineering (Z-Scores & CV)
    tweet_ts = posts["createdAt"].dt.tz_localize(None)
    stock_ts = stock["timestamp"].dt.tz_localize(None)
    stock_data = stock.reset_index(drop=True)
    
    final_rows = []
    for t_time in tweet_ts:
        t_min = t_time.floor("min")
        idx = stock_ts.searchsorted(t_min)
        
        if 10 <= idx < (len(stock_data) - 5) and stock_ts[idx] == t_min:
            pre = stock_data.iloc[idx-10 : idx]
            post = stock_data.iloc[idx+1 : idx+6]

            m_p, s_p = pre["close"].mean(), pre["close"].std()
            m_v, s_v = pre["volume"].mean(), pre["volume"].std()

            close_range = pre["close"].max() - pre["close"].min()

            row = {
                "close_delta_z": (pre["close"].iloc[-1] - pre["close"].iloc[0]) / (s_p + EPS),
                "volume_delta_z": (pre["volume"].iloc[-1] - pre["volume"].iloc[0]) / (s_v + EPS),
                "price_cv": s_p / (m_p + EPS),
                "volume_cv": s_v / (m_v + EPS),
                "close_position": (pre["close"].iloc[-1] - pre["close"].min()) / (close_range + EPS),
                "max_z_next5": ((post["close"] - m_p) / (s_p + EPS)).max(),
            }
            final_rows.append(row)
        else:
            final_rows.append({k: np.nan for k in ["close_delta_z"]})

    # F. Metadata & Float Sentiment Initialization
    features_df = pd.DataFrame(final_rows, index=posts.index)
    meta_df = pd.DataFrame({
        "whole_text": posts["whole_text"],
        "tweet_time": posts["createdAt"].dt.floor("min").dt.tz_localize(None),
        "mentions_tesla": posts["whole_text"].str.contains(TSLA_RE).astype(int),
        "is_reply": posts["isReply"].fillna(False).astype(int),
        "is_quote": posts["isQuote"].fillna(False).astype(int),
        "is_retweet": posts["isRetweet"].fillna(False).astype(int),
        "positive": 0.0, 
        "negative": 0.0, 
        "neutral": 0.0
    })

    # Merge, round, and remove incomplete windows
    result = pd.concat([meta_df, features_df], axis=1).dropna(subset=["close_delta_z", "max_z_next5"])
    result = result.round(3)

    # --- 2. FINAL METRICS ---
    end_combined_sum = len(result)
    end_trading_days = result["tweet_time"].dt.date.nunique()

    print(f"--- FINAL CLEANED STATS ---")
    print(f"Final Total Sum (Usable Events): {end_combined_sum}")
    print(f"Final Trading Days represented: {end_trading_days}")
    print(f"Net Retention: {(end_combined_sum / start_combined_sum) * 100:.2f}%")
    print("-" * 25)

    if save_csv:
        os.makedirs("data/cleaned", exist_ok=True)
        filename = f"data/cleaned/musk_events_k{k}_replies_{include_replies}.csv"
        result.to_csv(filename, index=False)
        print(f"File saved: {filename}")
    
    return result

# --- 5. EXECUTION ---
if __name__ == "__main__":
    # Ensure include_replies matches your previous successful run criteria
    final_df = run_pipeline(k=10, include_replies=True, save_csv=False)