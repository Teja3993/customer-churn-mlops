import pandas as pd
import re
from textblob import TextBlob

def clean_data(df):
    """Standardizes column names and handles missing values."""
    df = df.copy()
    df.columns = [re.sub('[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    return df.fillna("Unknown")

def add_sentiment(text):
    """Calculates NLP sentiment score."""
    return TextBlob(str(text)).sentiment.polarity

# This can be expanded as you add more preprocessing steps