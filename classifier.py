import newspaper
import pickle
import os.path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

keywords = []
keywords_rc = []

# Define the source, title, and category
source_url = 'http://www.telegraph.co.uk/'
source_name = 'The Telegraph'
category = 'education'

if not os.path.isfile(source_name + 'keywords.p'):
    # Building the corpus; momize_articles=False disables the caching
    paper = newspaper.build(source_url, memoize_articles=False)

    for article in paper.articles:

        if category in article.url:
            article_counter =+ 1
            article.download()
            article.parse()
            article.nlp(reference_corpus='coca_sampler.txt')

            keywords = keywords + article.keywords
            keywords_rc = keywords_rc + article.keywords_reference_corpus


    pickle.dump(keywords, open(source_name  + category + 'keywords.p', 'wb'))
    pickle.dump(keywords_rc, open(source_name  + category + 'keywords_rc.p', 'wb'))
else:
    keywords = pickle.load(open(source_name  + category + 'keywords.p', 'rb'))
    keywords_rc = pickle.load(open(source_name  + category + 'keywords_rc.p', 'rb'))


# Constructing two dataframes, one for each of the keyword extraction methods
keywords = Counter(keywords)
keywords_rc = Counter(keywords_rc)

df = pd.DataFrame.from_dict(keywords, orient='index').reset_index()
df = df.rename(columns={'index': 'Keyword', 0: 'Frequency (Articles)'})
df = df.sort_values(by=['Frequency (Articles)'], ascending=False)

df_rc = pd.DataFrame.from_dict(keywords_rc, orient='index').reset_index()
df_rc = df_rc.rename(columns={'index': 'Keyword', 0: 'Frequency (Articles)'})
df_rc = df_rc.sort_values(by=['Frequency (Articles)'], ascending=False)

# Data Visualization
how_many = 20
sns.set_style("whitegrid")
fig, axs = plt.subplots(ncols=2)
sns.barplot(x='Keyword', y='Frequency (Articles)', data=df[:how_many], ax=axs[0]).set_title('Keywords')
sns.barplot(x='Keyword', y='Frequency (Articles)', data=df_rc[:how_many], ax=axs[1]).set_title('Keywords COCA')
plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)
plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45)
plt.suptitle('{} ({})'.format(source_name, category))
fig.set_size_inches(20, 10)
plt.savefig(source_name + '.png', dpi=300)
