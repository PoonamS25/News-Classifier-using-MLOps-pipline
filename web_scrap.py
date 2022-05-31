


import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pymongo, nltk, newspaper
from newspaper import Article, Config
from newspaper import news_pool
from newspaper.utils import BeautifulSoup
conn = pymongo.MongoClient('mongodb://localhost:27017')
db = conn["database"]



url= "https://www.newsy.com/stories/commercial-companies-advance-space-exploration/"

seed_urls = ['https://inshorts.com/en/read/technology',
             'https://inshorts.com/en/read/sports',
             'https://inshorts.com/en/read/world']
config = Config()
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
config.browser_user_agent = user_agent
config.request_timeout = 15
gamespot = newspaper.build("https://inshorts.com/en/read/technology", memoize_articles = False)
bbc = newspaper.build("https://inshorts.com/en/read/sports", memoize_articles=False)
cnn = newspaper.build("https://inshorts.com/en/read/world", memoize_articles=False)

papers = [gamespot, bbc, cnn]
news_pool.set(papers, threads_per_source=4)

news_pool.join()
final_df = pd.DataFrame()
limit = 100

for source in papers:
    # temporary lists to store each element we want to extract
    print(source)
    list_title = []
    list_text = []
    list_source = []
    list_date = []
    list_category = []
    count = 0
    for article_extract in source.articles:
        article_extract.parse()
        article_extract.nlp()
        if count > limit:  # Lets have a limit, so it doesnt take too long when you're
            break  # running the code. NOTE: You may not want to use a limit

        # Appending the elements we want to extract

        list_title.append(article_extract.title)
        list_category.append(article_extract.source_url.split('/')[-1])
        list_text.append(article_extract.text)
        list_source.append(article_extract.source_url)
        list_date.append(article_extract.publish_date)
        # Update count
        count += 1

    temp_df = pd.DataFrame({'Title': list_title, 'Text': list_text,  'Published_date': list_date, 'Source': list_source, 'Category': list_category})
    # Append to the final DataFrame
    final_df = pd.concat([final_df,temp_df])

# From here you can export this to csv file
print(final_df)
final_df.to_csv('my_scraped_articles.csv')


#     for each_article in gamespot.articles:
#
#         each_article.download()
#         each_article.parse()
#         each_article.nlp()
#
#         temp_df = pd.DataFrame(columns=['Title', 'Authors', 'Text',
#                                     'Summary', 'published_date', 'Source'])
#
#         temp_df['Authors'] = each_article.authors
#
#         temp_df['Title'] = each_article.title
#         temp_df['Text'] = each_article.text
#         temp_df['Summary'] = each_article.summary
#         temp_df['published_date'] = each_article.publish_date
#         temp_df['Source'] = each_article.source_url
#     #final_df.append(temp_df, ignore_index=True)
#         final_df = pd.concat([final_df,temp_df])
#
# # From here you can export this Pandas DataFrame to a csv file
# print(final_df)
# final_df.to_csv('my_scraped_articles.csv')




