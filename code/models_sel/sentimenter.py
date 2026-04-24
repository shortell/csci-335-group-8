# adds tweet sentiment values to x inputs.
# uses vader and textblob instead of the network.

from textblob import TextBlob as tb #tuberculosis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # versus lol
import numpy as np

# df is a pandas dataframe
# output is a numpy array including textblob analysis & vader analysis
def sentimentalize(df):
    vader = SentimentIntensityAnalyzer()
    twts = df['whole_text'].fillna('').tolist()

    tb_score = [(tb(t).sentiment.polarity, tb(t).sentiment.subjectivity)
                 for t in twts]
    tb_arr    = np.array(tb_score)

    vd_score = [vader.polarity_scores(t) for t in twts]
    vd_arr    = np.array([[s['compound'], s['pos'], s['neu'], s['neg']]
                          for s in vd_score])
    return np.hstack([tb_arr, vd_arr])