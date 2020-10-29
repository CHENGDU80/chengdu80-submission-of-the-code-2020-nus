from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
import sparknlp
from sparknlp.pretrained import PretrainedPipeline

spark = sparknlp.start()


sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

parquet_file = "../../data/raw/data.parquet"
sdf = spark.read.parquet(parquet_file)
sdf.count()

sdf = sdf.withColumnRenamed("_1","title").withColumnRenamed("_2","text")
sdf.printSchema()

sdf.createOrReplaceTempView("news") 

sql_query = """
    SELECT *
    FROM news
    LIMIT 5
"""

spark.sql(sql_query).show()

sql_query = """
    SELECT SUBSTRING(title, 74, 10) AS date, text
    FROM news
"""

sdf = spark.sql(sql_query)
sdf.show(5)

news_df = sdf.toPandas()

news_df.shape

news_df.to_csv("../../data/raw/news.csv", index=False)

# Usual Imports
import pandas as pd
import datetime
import re
import numpy as np
from pprint import pprint
import pickle

# NLP Imports
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import TfidfModel, LdaModel, CoherenceModel, LdaMulticore
from gensim import corpora, models, similarities, matutils
import spacy
import nltk
nltk.download('stopwords')

# Visualization Imports
import matplotlib.pyplot as plt
%matplotlib inline
import pyLDAvis
import pyLDAvis.gensim
from wordcloud import WordCloud

# Misc Imports
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

filepath = "../../data/raw/news.csv"

df = pd.read_csv(filepath)

df['title'] = df['text'].str.extract("(?<=-- )(.+)(?=\\n-- B)")

df.head()

df.dropna(inplace=True)

# Preprocessing
def preprocess(article):
    
    # Tokenize
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
    
    data_words = list(sent_to_words(article)) 
    
    # Remove Stopwords & Short Words
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words and len(word) > 2] for doc in texts]
    
    data_words_nostops = remove_stopwords(data_words)
    
    # Bigrams
    bigram_config = gensim.models.Phrases(data_words_nostops, min_count=5, threshold=100) 
    bigram_mod = gensim.models.phrases.Phraser(bigram_config)
 
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]
    
    bigrams_list = make_bigrams(data_words_nostops)
    
    # Lemmatization
    def lemmatization(texts):
    
        nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        allowed_postags = ['NOUN', 'ADJ']
        texts_out = []
    
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
            
        return texts_out
    
    data_lemmatized = lemmatization(bigrams_list)
    
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]
    
    return corpus, id2word, data_lemmatized

bow, id2word, data_lemmatized = preprocess(df['text'])

# Save Data
with open('../../data/pickle/bow.pickle', 'wb') as file:
    pickle.dump(bow, file)

with open('../../data/pickle/id2word.pickle', 'wb') as file:
    pickle.dump(id2word, file)

with open('../../data/pickle/data_lemmatized.pickle', 'wb') as file:
    pickle.dump(data_lemmatized, file)

# # Load Data
# bow = pickle.load(open('../../data/pickle/bow.pickle', 'rb'))
# id2word = pickle.load(open('../../data/pickle/id2word.pickle', 'rb'))
# data_lemmatized = pickle.load(open('../../data/pickle/data_lemmatized.pickle', 'rb'))

# Mallet
mallet_path = '~/mallet-2.0.6/bin/mallet'
mallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=bow, num_topics=30, id2word=id2word)

# Show Topics
pprint(mallet.show_topics(formatted=False))

with open('../../data/pickle/mallet.pickle', 'wb') as file:
    pickle.dump(mallet, file)
    
# mallet = pickle.load(open('../../data/pickle/mallet.pickle', 'rb'))

import sys

orig_stdout = sys.stdout

with open('../../data/processed/topic_words.txt', 'w') as f:
    sys.stdout = f
    pprint(mallet.show_topics(num_topics=-1, num_words=75, formatted=False))
    f.close()

sys.stdout = orig_stdout
    
pprint(mallet.show_topics(num_topics=-1, num_words=75, formatted=False))

converted_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(mallet)

def get_doc_topics(model, bow):
    
    doc_topics = pd.DataFrame()
    
    corpus = model[bow]
    
    for idx, doc in enumerate(corpus):
        doc_topics = pd.concat([doc_topics, pd.DataFrame(doc)[1]], axis=1)
        
        if idx % 1000 == 0:
            print(idx, "/", len(bow), "completed")
        
    return doc_topics

document_scores = get_doc_topics(converted_model, bow)

document_scores.head()

document_scores.columns = list(range(len(bow)))

doc_topics_df = document_scores.T

doc_topics_df.columns = ["topic_" + str(x) for x in list(range(doc_topics_df.shape[1]))]

doc_topics_df.head()

# Def topic mapping
num = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,19,20,21,25,28]
topics = ['topic_' + str(x) for x in num]

topics_df = doc_topics_df[topics]

topics_df.head()

topics_df.rename(columns={
    "topic_2": "defence",
    "topic_3": "staples",
    "topic_4": "finance",
    "topic_5": "sports",
    "topic_6": "real_estate",
    "topic_7": "life_and_entertainment",
    "topic_8": "industrial",
    "topic_9": "politics",
    "topic_10": "energy",
    "topic_11": "economy",
    "topic_12": "staples_2",
    "topic_13": "economy_2",
    "topic_14": "healthcare",
    "topic_15": "energy_2",
    "topic_19": "finance_2",
    "topic_20": "consumer_discretionary",
    "topic_21": "governance",
    "topic_25": "info_tech",
    "topic_28": "weather_and_climate"
}, inplace=True)

topics_df.head()

topics_df['main_topic'] = topics_df.idxmax(axis=1)

topics_df.head()

classified_docs = pd.merge(left=df, right=topics_df, left_on=df.index, right_on=topics_df.index, how='inner').drop(columns=["key_0"])

classified_docs.head()

classified_docs.to_csv("../../data/processed/doc_topics.csv", index=False)

mini_docs = classified_docs[['date', 'text', 'title', 'main_topic']]

mini_docs.head()

import re
patt = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"

test = [re.findall(patt, x)[0][0] for x in mini_docs['text']]

mini_docs['url'] = test

mini_docs.head()

mini_docs = mini_docs[['date', 'title', 'main_topic', 'url']]

mini_docs.to_csv("../../data/processed/news_articles.csv", index=False, encoding="UTF-8")

