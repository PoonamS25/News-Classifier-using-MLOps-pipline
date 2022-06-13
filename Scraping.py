
import requests
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pymongo, nltk, newspaper
from newspaper import Config
from gensim import corpora, models
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# We need this dataset in order to use the tokenizer
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.decomposition import NMF
# Also download the list of stopwords to filter out
#nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
LDA_topics_theme = ['Technology', 'Sports', 'Education', 'Business', 'Entertainment']
NMF_topics_theme = ['Politics/National','Sports/Cricket','Entertainment/Movie','Technology','Health/Pandemic']
topic_list = LDA_topics_theme + NMF_topics_theme

vectorizer = CountVectorizer(analyzer='word',
                             min_df=3,                        # minimum required occurences of a word
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             max_features=5000,             # max number of unique words. Build a vocabulary that only consider the top max_features ordered by term frequency across the corpus
                            )

lda_model = LatentDirichletAllocation(n_components=5, # Number of topics
                                      learning_method='online',
                                      random_state=0,
                                      n_jobs = -1  # Use all available CPUs
                                     )

def process_text(text):
    # Make all the strings lowercase and remove non alphabetic characters
    text = re.sub('[^A-Za-z]', ' ', text.lower())

    # Tokenize the text; this is, separate every sentence into a list of words
    # Since the text is already split into sentences you don't have to call sent_tokenize
    tokenized_text = word_tokenize(text)

    # Remove the stopwords and stem each word to its root
    clean_text = [
        stemmer.stem(word) for word in tokenized_text
        if word not in stopwords.words('english')
    ]

    # Remember, this final output is a list of words
    return clean_text


def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=5):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
#         print(feature_name)
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()

        if (max_value - min_value != 0):
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        else:
            result[feature_name] = 0
    return result


def label_theme(row):
    if row['dominant_topic'] > len(topic_list) or row['dominant_topic'] < 0:
        return ""
    return topic_list[int(row['dominant_topic'])]


config = Config()
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
config.browser_user_agent = user_agent
config.request_timeout = 50
gamespot = newspaper.build("https://inshorts.com/en/read/technology", memoize_articles = False, config = config)
bbc = newspaper.build("https://www.bbc.com/news", memoize_articles=False)

papers = [gamespot, bbc]
final_df = pd.DataFrame()
counter = 1
for each_article in gamespot.articles:

    each_article.download()
    each_article.parse()
    each_article.nlp()

    temp_df = pd.DataFrame(columns=['ID', 'Authors', 'Text'])

    temp_df['Authors'] = each_article.authors
    #print(each_article.authors)
    #temp_df['Title'] = each_article.title
    temp_df['ID'] = counter
    temp_df['Text'] = each_article.text
    #temp_df['Summary'] = each_article.summary
    #temp_df['Published_date'] = each_article.publish_date
    #temp_df['Source'] = each_article.source_url

    counter +=1
    final_df = pd.concat([final_df, temp_df])

# From here you can export this Pandas DataFrame to a csv file
print(final_df)
#final_df.to_csv('my_scraped_articles.csv')





df_clean = pd.DataFrame(final_df['Text'].apply(lambda x: clean_text(x)))
df_clean = pd.concat([df_clean, final_df['ID']], axis=1)
df_clean.to_csv('file.csv',index=False)
df_clean = pd.read_csv('file.csv')
print(df_clean)

data = df_clean[['ID','Text']]
data = data[data['ID'].notnull()]
data["ID"] = data['ID'].astype('int64')
data.dropna(subset=['Text'],how='all',inplace=True)
data = data[data['Text'].map(len) > 20]

data.reset_index(drop=True, inplace=True)
print(data)

data_vectorized = vectorizer.fit_transform(data['Text'])

lda_output = lda_model.fit_transform(data_vectorized)


topic_keywords = show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=5)

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
print(df_topic_keywords)


df_topic_keywords['topic_theme'] = LDA_topics_theme
df_topic_keywords.set_index('topic_theme', inplace=True)
print(df_topic_keywords.T)

n_features = 2000
n_components = 5
n_top_words = 5
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(data['Text'].values.astype(str))
# nmf = NMF(n_components=n_components, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)

nmf = NMF(n_components=n_components, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)

def show_topics(vectorizer=tfidf_vectorizer, nmf_model=nmf, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in nmf_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=tfidf_vectorizer, nmf_model=nmf, n_words=20)
# Topic - Keywords Dataframe
df_topic_keywords1 = pd.DataFrame(topic_keywords)
df_topic_keywords1.columns = ['Word '+str(i) for i in range(df_topic_keywords1.shape[1])]
df_topic_keywords1.index = ['Topic '+str(i) for i in range(df_topic_keywords1.shape[0])]


df_topic_keywords1['topic_theme'] = NMF_topics_theme
df_topic_keywords1.set_index('topic_theme', inplace=True)
nmf_output = nmf.transform(tfidf)


v = lda_output
v = v*100
#len(v)

from tqdm import tqdm
s = []
for i in tqdm(range(len(nmf_output))):
    s1 = nmf_output[i]/sum(nmf_output[i])
    s.append(s1)
nmf_output = np.array(s)
d = nmf_output
d = d * 100
len(d)


LDA_df = pd.DataFrame(v,columns=df_topic_keywords.T.columns)
NMF_df = pd.DataFrame(d,columns=df_topic_keywords1.T.columns)

LDA_normalized = normalize(LDA_df)
NMF_normalized = normalize(NMF_df)
print(LDA_normalized)

LDANMF = pd.concat([NMF_normalized,LDA_normalized],axis=1)
LDANMF.head(23)

def computeConfidence(similarityList):
    similarScores = set(similarityList)
    highest = max(similarScores)

    similarScores.remove(highest)
    if (len(similarScores) == 0):
        return 0

    secondHighest = max(similarScores)

    return (highest - secondHighest) / (highest)


dominant_topic = np.argmax(LDANMF.values, axis=1)

LDANMF['confidence'] = LDANMF.apply (lambda row: computeConfidence(row), axis=1)
LDANMF['dominant_topic'] = dominant_topic
final = pd.concat([data,LDANMF[['dominant_topic','confidence']]],axis=1)
print(final)

final['dominant_topic'] = final.apply (lambda row: label_theme(row), axis=1)
final.to_csv('news_category.csv')
print(final)