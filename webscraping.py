import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pymongo, nltk
from collections import deque

seed_urls = ['https://inshorts.com/en/read/technology',
             'https://inshorts.com/en/read/sports',
             'https://inshorts.com/en/read/world']


class CircularBuffer(deque):
    def __init__(self, size=0):
        super(CircularBuffer, self).__init__(maxlen=size)
    @property
    def average(self):  # TODO: Make type check for integer or floats
        print(self)


def build_dataset(seed_urls):
    news_data = []
    print("indside build function")
    for url in seed_urls:
        news_category = url.split('/')[-1]
        data = requests.get(url)
        soup = BeautifulSoup(data.content, 'html.parser')

        news_articles = [{'news_headline': headline.find('span', attrs={"itemprop": "headline"}).string,
                          'news_article': article.find('div', attrs={"itemprop": "articleBody"}).string,
                          'news_category': news_category}
                         for headline, article in zip(soup.find_all('div', class_=["news-card-title news-right-box"]),
                                 soup.find_all('div', class_=["news-card-content news-right-box"]))]
        news_data.extend(news_articles)
        #db.myTable.insert_many(news_articles)


    df =  pd.DataFrame(news_data)
    df = df[['news_headline', 'news_article', 'news_category']]
    return df


news_df = build_dataset(seed_urls)
print(news_df.head(10))
print(news_df.news_category.value_counts())



news_df.reset_index(inplace=True)
data_dict = news_df.to_dict()
print(data_dict)
conn = pymongo.MongoClient('mongodb://localhost:27017')
db = conn["database"]
db.myTable.insert_many(news_df.to_dict("records"))
print(f'inserted {len(news_df)} articles')
col = db["myTable"]

x = col.find()
cb = CircularBuffer(size=10)
for data in range(20):
    #print(data)
    cb.append(data)
    cb.average







