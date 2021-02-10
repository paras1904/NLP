import pandas as pd
import numpy as np

df = pd.read_csv('sentiment_analysis.csv',encoding='latin-1')
df = df[['ï»¿text','label']]
df = df.rename(columns={'ï»¿text':'text'})

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

print(sid.polarity_scores(df.loc[0]['text']))
df['score'] = df['text'].apply(lambda text: sid.polarity_scores(text))
df['compound'] = df['score'].apply(lambda score_dict: score_dict['compound'])
df['comp_score'] = df['compound'].apply(lambda c:'pos' if c>=0 else 'neg')
print(df.head())