import pandas as pd
import re
import html
import os


# ── helpers ────────────────────────────────────────────────────────────────────

URL_RE = re.compile(r"http\S+")
AT_RE  = re.compile(r"@\w+")          # removes every @handle including mid-text
WS_RE  = re.compile(r"\s+")
RT_RE  = re.compile(r"^RT\s+")        # strips leading "RT " marker

def _vectorized_clean(series: pd.Series) -> pd.Series:
    """
    Clean a text Series (vectorized):
      - html-unescape &amp; etc.
      - strip all URLs
      - strip all @handles (anywhere in the text)
      - strip leading 'RT ' retweet marker
      - collapse whitespace
    """
    s = series.fillna("")
    # html unescape only if needed
    if s.str.contains("&", regex=False).any():
        s = s.apply(html.unescape)
    s = s.str.replace(URL_RE, "", regex=True)
    s = s.str.replace(AT_RE,  "", regex=True)
    s = s.str.replace(RT_RE,  "", regex=True)
    s = s.str.replace(WS_RE,  " ", regex=True).str.strip()
    return s


def _to_naive_eastern(ts: pd.Series) -> pd.Series:
    """
    Normalize any timestamp Series to tz-naive America/New_York wall-clock time.
    This is used exclusively for the stock-lookup search so both sides of the
    searchsorted are in the same dtype (no tz vs tz mismatch).
    """
    if ts.dt.tz is not None:
        return ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return ts


# ── pipeline steps ─────────────────────────────────────────────────────────────

def load_data(posts_path=None, quotes_path=None, stock_path=None):
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if posts_path  is None: posts_path  = os.path.join(base, "data", "all_musk_posts.csv")
    if quotes_path is None: quotes_path = os.path.join(base, "data", "musk_quote_tweets.csv")
    if stock_path  is None: stock_path  = os.path.join(base, "data", "TSLA_1min_market_hours_2016_2025.csv")

    posts  = pd.read_csv(posts_path,  low_memory=False)
    quotes = pd.read_csv(quotes_path)
    stock  = pd.read_csv(stock_path)
    return posts, quotes, stock


def localize_timezones(posts, quotes, stock):
    """
    Convert all timestamps to tz-aware America/New_York.
    Stock data comes in as tz-naive Eastern (Alpaca API) and is localized.
    Posts/quotes are stored as UTC strings and are converted.
    """
    posts  = posts.copy()
    quotes = quotes.copy()
    stock  = stock.copy()

    # Posts: UTC string -> tz-aware Eastern
    posts["createdAt"] = (
        pd.to_datetime(posts["createdAt"], utc=True)
        .dt.tz_convert("America/New_York")
    )

    # Quotes: both timestamp columns
    for col in ("orig_tweet_created_at", "musk_quote_created_at"):
        quotes[col] = (
            pd.to_datetime(quotes[col], utc=True)
            .dt.tz_convert("America/New_York")
        )

    # Stock: Alpaca delivers tz-naive Eastern wall-clock times; localize directly.
    ts = pd.to_datetime(stock["timestamp"])
    if ts.dt.tz is None:
        stock["timestamp"] = ts.dt.tz_localize(
            "America/New_York", nonexistent="shift_forward", ambiguous="NaT"
        )
    else:
        stock["timestamp"] = ts.dt.tz_convert("America/New_York")

    return posts, quotes, stock


def drop_bad_rows(posts, quotes):
    """
    Drop any post that is missing id or createdAt — these rows cannot be
    matched to stock data or used for downstream tasks.
    Also drop any quotes missing their musk_tweet_id or timestamp.
    """
    posts  = posts.dropna(subset=["id", "createdAt"]).copy()
    quotes = quotes.dropna(subset=["musk_tweet_id", "musk_quote_created_at"]).copy()
    # id must be a valid integer (no float NaN sneaking through)
    posts["id"] = posts["id"].astype("int64")
    quotes["musk_tweet_id"] = quotes["musk_tweet_id"].astype("int64")
    return posts, quotes


def filter_to_overlapping_date_range(posts, quotes, stock):
    date_min = stock["timestamp"].min()
    date_max = stock["timestamp"].max()
    posts  = posts[posts["createdAt"].between(date_min, date_max)]
    quotes = quotes[quotes["musk_quote_created_at"].between(date_min, date_max)]
    return posts, quotes, stock


def filter_to_market_hours_only(posts, quotes, stock):
    def between_market(df, col, start, end):
        idx  = pd.DatetimeIndex(df[col]).tz_convert("America/New_York")
        mask = idx.indexer_between_time(start, end)
        return df.iloc[mask]

    # Stock: full trading session kept for the 15-min window lookup
    stock  = between_market(stock,  "timestamp",             "09:30", "16:00")
    # Tweets: capped at 15:45 so every tweet has at least 15 min left in session
    posts  = between_market(posts,  "createdAt",             "09:30", "15:45")
    quotes = between_market(quotes, "musk_quote_created_at", "09:30", "15:45")

    return (
        posts.reset_index(drop=True),
        quotes.reset_index(drop=True),
        stock.reset_index(drop=True),
    )


def filter_urls(posts, quotes):
    """
    Remove quote-tweet rows where both Musk's comment AND the original tweet
    are empty after cleaning (i.e. the tweet was purely a URL).
    Plain posts are all kept regardless.
    """
    quotes = quotes.copy()
    quotes["_clean_musk"] = _vectorized_clean(quotes["musk_quote_tweet"])
    quotes["_clean_orig"] = _vectorized_clean(quotes["orig_tweet_text"])

    quotes = quotes[
        (quotes["_clean_musk"].str.len() > 0) |
        (quotes["_clean_orig"].str.len() > 0)
    ].drop(columns=["_clean_musk", "_clean_orig"])

    quote_ids = set(quotes["musk_tweet_id"])
    non_quote_mask = posts["isQuote"] != True
    quote_mask     = (posts["isQuote"] == True) & posts["id"].isin(quote_ids)
    posts = (
        pd.concat([posts[non_quote_mask], posts[quote_mask]], ignore_index=True)
        .sort_values("createdAt")
        .drop_duplicates(subset="id")
    )
    return posts, quotes


def build_clean_text_and_drop_short(posts, quotes, k: int = 15):
    """
    Build cleanText per post:
      - Non-quote tweets: clean(fullText)
      - Quote tweets: clean(originalTweet) + " " + clean(muskComment)
      - Replies/retweets: prepend context text if available in the dataset

    Posts whose cleaned Musk text is shorter than k words are dropped.
    """
    quotes = quotes.copy()
    quotes["cleanMuskText"] = _vectorized_clean(quotes["musk_quote_tweet"])
    quotes["cleanOrigText"] = _vectorized_clean(quotes["orig_tweet_text"])
    quotes["combinedText"]  = (
        quotes["cleanOrigText"] + " " + quotes["cleanMuskText"]
    ).str.strip()
    quote_text_map = quotes.set_index("musk_tweet_id")["combinedText"]

    posts = posts.copy()
    clean_full = _vectorized_clean(posts["fullText"])
    posts["cleanText"] = clean_full

    # Quote tweets: replace with combined original + Musk comment
    is_quote = posts["isQuote"] == True
    posts.loc[is_quote, "cleanText"] = (
        posts.loc[is_quote, "id"]
            .map(quote_text_map)
            .fillna(clean_full[is_quote])
    )

    # Replies: prepend replied-to text if the column exists in the dataset
    if "inReplyToText" in posts.columns:
        is_reply = posts["isReply"] == True
        reply_ctx = _vectorized_clean(posts.loc[is_reply, "inReplyToText"])
        posts.loc[is_reply, "cleanText"] = (
            reply_ctx + " " + posts.loc[is_reply, "cleanText"]
        ).str.strip()

    # Retweets: prepend original tweet text if available
    if "retweetedText" in posts.columns:
        is_rt = posts["isRetweet"] == True
        rt_ctx = _vectorized_clean(posts.loc[is_rt, "retweetedText"])
        posts.loc[is_rt, "cleanText"] = (
            rt_ctx + " " + posts.loc[is_rt, "cleanText"]
        ).str.strip()

    # Drop short posts (word count based on Musk's own text only)
    word_counts = clean_full.str.split().str.len().fillna(0)
    posts = posts[word_counts >= k]

    return posts, quotes


def enforce_15m_tweet_isolation(posts, quotes):
    """
    Keep only tweets that have no other Musk tweet within 15 minutes before
    or after — ensures each tweet's stock window is clean.
    """
    posts = posts.sort_values("createdAt").copy()
    t = posts["createdAt"]
    gap_before = t.diff().dt.total_seconds().div(60).fillna(float("inf"))
    gap_after  = t.diff(periods=-1).dt.total_seconds().div(-60).fillna(float("inf"))

    posts  = posts[(gap_before > 15.0) & (gap_after > 15.0)].copy()
    quotes = quotes[quotes["musk_tweet_id"].isin(posts["id"])].copy()
    return posts, quotes


def align_to_active_trading_days(posts, quotes, stock):
    stock_days       = pd.DatetimeIndex(stock["timestamp"]).normalize()
    post_days        = pd.DatetimeIndex(posts["createdAt"]).normalize()
    quote_days       = pd.DatetimeIndex(quotes["musk_quote_created_at"]).normalize()

    trading_day_set  = set(stock_days)
    active_tweet_set = set(post_days)

    posts   = posts[post_days.isin(trading_day_set)]
    quotes  = quotes[quote_days.isin(trading_day_set)]
    stock   = stock[stock_days.isin(active_tweet_set)]
    return posts, quotes, stock


# ── output builder ─────────────────────────────────────────────────────────────

# TSLA candle columns to include for each minute offset
_STOCK_COLS = ["close", "volume", "trade_count"]

# Minute offsets after the tweet to capture (0 = bar at tweet time as baseline)
_OFFSETS = [0, 2, 4, 8, 16]


def build_output_csv(posts: pd.DataFrame, quotes: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:
    """
    Produce one flat CSV row per tweet with:

    Columns (in order):
      row_id            – unique integer row identifier (for downstream vector matching)
      tweet_id          – Musk's tweet ID (int64)
      tweet_timestamp   – tweet time, tz-aware Eastern (ISO string in CSV)
      cleanText         – fully cleaned text (tweet + quoted/replied-to text if applicable)
      likeCount         – engagement: likes
      retweetCount      – engagement: retweets
      replyCount        – engagement: replies
      stock_t{2,4,8,16}_timestamp  – 1-min bar timestamp (tz-naive Eastern in CSV)
      stock_t{2,4,8,16}_close/volume/trade_count

    Stock alignment:
      The window starts at the FIRST 1-minute bar whose timestamp is >= the
      tweet time (both compared as tz-naive Eastern wall-clock to avoid any
      tz-aware vs tz-naive dtype mismatch). Only bars at offsets +2, +4, +8,
      and +16 minutes are captured. Bars outside the trading session are NaN.
    """
    posts  = posts.copy().reset_index(drop=True)
    quotes = quotes.copy()
    stock  = stock.copy()

    # ── 1. Guarantee id / createdAt are non-null ──────────────────────────────
    posts = posts.dropna(subset=["id", "createdAt"]).reset_index(drop=True)
    posts["id"] = posts["id"].astype("int64")

    # ── 2. Make sure cleanText reflects the combined text for quote tweets ─────
    if "combinedText" in quotes.columns:
        quote_text_map = quotes.set_index("musk_tweet_id")["combinedText"]
    else:
        quote_text_map = quotes.set_index("musk_tweet_id")["musk_quote_tweet"]

    is_quote = posts["isQuote"] == True
    posts.loc[is_quote, "cleanText"] = (
        posts.loc[is_quote, "id"]
             .map(quote_text_map)
             .fillna(posts.loc[is_quote, "cleanText"])
    )

    # ── 3. Build tz-naive Eastern stock index for alignment ───────────────────
    stock_sorted  = stock.sort_values("timestamp").reset_index(drop=True)
    # Convert to tz-naive Eastern wall-clock for searchsorted comparison
    stock_ts_naive = _to_naive_eastern(stock_sorted["timestamp"])

    # Tweet times as tz-naive Eastern wall-clock (same dtype as stock_ts_naive)
    tweet_ts_naive = _to_naive_eastern(posts["createdAt"])

    # ── 4. For each tweet, find bars at exactly +2, +4, +8, +16 min offsets ───
    # Anchor on the floored tweet minute so that the displayed timestamps are
    # always exactly N minutes apart (e.g. tweet=13:21 → t+2=13:23, t+4=13:25).
    stock_rows = []
    for tweet_time_naive in tweet_ts_naive:
        tweet_min = tweet_time_naive.floor("min")  # floor to minute boundary
        row_dict: dict = {}
        for offset in _OFFSETS:   # t=2, 4, 8, 16 minutes after tweet
            target = tweet_min + pd.Timedelta(minutes=offset)
            idx    = stock_ts_naive.searchsorted(target, side="left")
            prefix = f"stock_t{offset}_"
            if idx < len(stock_sorted):
                bar = stock_sorted.iloc[idx]
                row_dict[prefix + "timestamp"] = bar["timestamp"]
                for col in _STOCK_COLS:
                    row_dict[prefix + col] = bar[col]
            else:
                row_dict[prefix + "timestamp"] = pd.NaT
                for col in _STOCK_COLS:
                    row_dict[prefix + col] = float("nan")
        stock_rows.append(row_dict)

    stock_df = pd.DataFrame(stock_rows, index=posts.index)

    # ── 5. Assemble final DataFrame with slim metadata only ───────────────────
    out = pd.DataFrame({
        "row_id":          range(len(posts)),        # unique int for vector matching
        "tweet_id":        posts["id"].values,
        "tweet_timestamp": posts["createdAt"].values,
        "cleanText":       posts["cleanText"].values,
        "likeCount":       posts.get("likeCount",    pd.Series(dtype=float)).reindex(posts.index).values,
        "retweetCount":    posts.get("retweetCount", pd.Series(dtype=float)).reindex(posts.index).values,
        "replyCount":      posts.get("replyCount",   pd.Series(dtype=float)).reindex(posts.index).values,
    })

    result = pd.concat([out, stock_df.reset_index(drop=True)], axis=1)

    # ── 6. Final integrity check: drop rows missing any required field ────────
    required = ["tweet_id", "tweet_timestamp", "likeCount", "retweetCount", "replyCount"]
    result = result.dropna(subset=required).reset_index(drop=True)
    # Re-assign row_id to be sequential after any final drops
    result["row_id"] = range(len(result))

    # ── 7. Normalize all timestamps: tz-naive Eastern wall-clock, minute precision
    # tweet_timestamp loses its tz via numpy .values (becomes naive UTC); stock
    # timestamps are already tz-aware Eastern. Both are converted to Eastern then
    # stripped of tz info so every column is a plain YYYY-MM-DD HH:MM string —
    # no offset suffix needed since all data is already filtered to market hours.
    ts_cols = ["tweet_timestamp"] + [f"stock_t{o}_timestamp" for o in _OFFSETS]
    for col in ts_cols:
        if col not in result.columns:
            continue
        s = pd.to_datetime(result[col])
        if s.dt.tz is None:
            # tz was stripped by numpy — the underlying value is UTC
            s = s.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
        else:
            s = s.dt.tz_convert("America/New_York")
        result[col] = s.dt.floor("min").dt.tz_localize(None)  # drop tz offset

    return result


# ── main ───────────────────────────────────────────────────────────────────────

def run_pipeline(k: int = 15, save_csv: bool = True):
    posts, quotes, stock = load_data()
    posts, quotes, stock = localize_timezones(posts, quotes, stock)
    posts, quotes        = drop_bad_rows(posts, quotes)
    posts, quotes, stock = filter_to_overlapping_date_range(posts, quotes, stock)
    posts, quotes, stock = filter_to_market_hours_only(posts, quotes, stock)
    posts, quotes        = filter_urls(posts, quotes)
    posts, quotes        = build_clean_text_and_drop_short(posts, quotes, k=k)
    posts, quotes        = enforce_15m_tweet_isolation(posts, quotes)
    posts, quotes, stock = align_to_active_trading_days(posts, quotes, stock)

    n_samples = len(posts)
    print(f"Total samples (k={k}): {n_samples}")

    output = build_output_csv(posts, quotes, stock)

    if save_csv:
        base     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir  = os.path.join(base, "data", "cleaned")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "pipeline_output.csv")
        output.to_csv(out_path, index=False)
        print(f"Saved {len(output)} rows -> {out_path}")
        print(f"Columns ({len(output.columns)}): {list(output.columns)}")

    return posts, quotes, stock, output


if __name__ == "__main__":
    run_pipeline(k=10, save_csv=True)