import pandas as pd

# For now I decided to work with the posts and tesla stock until
# we come up with a plan to create a pipeline that could incorporate
# the quoted tweet context
all_musk_posts = pd.read_csv("../data/original/all_musk_posts.csv", low_memory=False)
tesla_stock    = pd.read_csv("../data/original/TSLA_1min_market_hours_2016_2025.csv")

# Keep only relevant columns
all_musk_posts = all_musk_posts[['fullText', 'createdAt']]
tesla_stock    = tesla_stock[['timestamp','open', 'close', 'volume']]

# Convert to datetime so Python understands them as actual time, not just strings
all_musk_posts['createdAt']  = pd.to_datetime(all_musk_posts['createdAt'], utc=True)
tesla_stock['timestamp']     = pd.to_datetime(tesla_stock['timestamp'], utc=True)

print("all_musk_posts shape:", all_musk_posts.shape)
print("tesla_stock shape:", tesla_stock.shape)
print("\nEarliest tweet:", all_musk_posts['createdAt'].min())
print("Latest tweet:",    all_musk_posts['createdAt'].max())
print("\nEarliest stock:", tesla_stock['timestamp'].min())
print("Latest stock:",    tesla_stock['timestamp'].max())

# Drop tweets that fall outside the stock data range
# Any tweet before 2016 or after 2025 can never be matched to stock data
stock_start = tesla_stock['timestamp'].min()
stock_end   = tesla_stock['timestamp'].max()

all_musk_posts = all_musk_posts[
    (all_musk_posts['createdAt'] >= stock_start) &
    (all_musk_posts['createdAt'] <= stock_end)
]


earliest_time = all_musk_posts['createdAt'].min()
print(earliest_time)

# Use only the tesla stock from the begining of the posts
# tesla_stock = tesla_stock[
#     (tesla_stock['timestamp'] >= earliest_time)
# ]

print("\nTweets after filtering:", all_musk_posts.shape)
print("Earliest tweet now:", all_musk_posts['createdAt'].min())
print("Lastest tweet now:", all_musk_posts['createdAt'].max())

print("\nTesla stock after filtering:", tesla_stock.shape)
print("Earliest stock timestamp now: ", tesla_stock['timestamp'].min())
print("Latest stock timestamp now: ", tesla_stock['timestamp'].max())