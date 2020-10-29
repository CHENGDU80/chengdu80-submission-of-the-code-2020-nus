import pandas as pd
import numpy as np

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

doc_topics = pd.read_csv("../../data/processed/doc_topics.csv")
doc_topics.head()

import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize, RegexpTokenizer

def get_word_sentiment(text):
    
    tokenized_text = nltk.word_tokenize(text)
    analyzer = SentimentIntensityAnalyzer()
    
    pos_word_list=[]
    neu_word_list=[]
    neg_word_list=[]
    
    for word in tokenized_text:
            score = analyzer.polarity_scores(word)['compound']
        
            if score >= 0.1:
                pos_word_list.append((word, score))
            elif score <= -0.1:
                neg_word_list.append((word, score))
            else:
                neu_word_list.append((word, score))
                
    print('Positive:',pos_word_list)        
    print('Neutral:',neu_word_list)
    print('Negative:',neg_word_list)
    
    print("Overall Sentiment:", analyzer.polarity_scores(text)['compound'])

doc_topics['sentiment_text'] = [x['compound'] for x in doc_topics['text'].apply(sia.polarity_scores)]

doc_topics.head()

doc_topics.to_csv("../../data/processed/doc_topics_sentiments.csv", index=False)

transform_df = doc_topics[['date', 'main_topic', 'sentiment_text']]

transform_df.head()

daily_sentiments = transform_df.groupby(["date", "main_topic"], as_index=False).mean()

daily_sentiments.head()

sentiments_df = daily_sentiments.pivot(index='date', columns='main_topic', values='sentiment_text').fillna(0)

sentiments_df.head()

sentiments_df.to_csv("../../data/processed/sentiments.csv", index=True)

url_df = pd.read_csv("../../data/processed/news_articles.csv")

url_df.head()

article_sentiments = pd.merge(url_df, doc_topics, how="inner", on=["date", "title"])[['date', 'title', 'main_topic_x', 'url', 'sentiment_text']]

article_sentiments.to_csv("../../data/processed/article_sentiments.csv", index=False)