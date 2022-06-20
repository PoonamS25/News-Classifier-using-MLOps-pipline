# News-Classifier-using-MLOps-pipline
The news category classification aims to recognize and categorize different news articles based on content/information type. The automatic news classification plays a vital role in processing a massive amount of news content. It can classify and label the news articles by analyzing the content (i.e., extracting feature values) to quickly access what they are interested in, allowing efficient and speedy news dissemination.

This is a classification model that is able to predict the category of a given news article, a web scraping method that gets the latest news from the online newspapers.

MLOps, Machine learning and Operations is more to production-grade machine learning systems than designing learning algorithms and writing code. Being able to select and design the most optimal architecture for this project is often what bridges the gap between machine learning and operations
In common architectural patterns for MLOps, architectural changes occur at the ML stage as well as the Ops stage, where you can have various development and deployment patterns that depend on the problem and the data.
Following is Orchestrated pull-based MLOps general training architecture

![image](https://user-images.githubusercontent.com/101706028/166643884-e4616803-1455-4adf-a6ea-390a561c2c77.png)
This is the training architecture for scenarios where we have to retrain our model at scheduled intervals. Data is waiting in the warehouse and a workflow orchestration tool is used to schedule the extraction and processing, as well as the retraining of the model on fresh data.
Architectural overview

![image](https://user-images.githubusercontent.com/101706028/166643940-aa979357-1e2f-400c-8bd6-da681678b20b.png)

Data 	Scraping
The fundamental part of any machine learning workflow is data. Collecting good data sets has a huge impact on the quality and performance of the ML model. We need to write a web-scraping script that gathers the news from the online newspaper, gets into their link and scrapes the news body paragraphs continuously.

Data 	extraction and cleaning
We select and integrate the relevant data from various data sources for the ML task
Text Cleaning
Before creating any feature from the raw text, we must perform a cleaning process to ensure no distortions will introduced to the model. Some steps for text cleaning:

Special 	character cleaning: special characters such as “\n” 	double quotes must 	be removed from the text.
Upcase/downcase:“Book”and“book”to be 	the 	same word and have the same predicting power. For that reason we 	must 	to down cased every word.
Punctuation signs: characters such as “?”,“!”,“;” must 	be removed.
Possessive pronouns: in addition, we would expect that “Trump” and “Trump’s”  had the same predicting power.
Stemming or Lemmatization : stemming is the process of 	 reducing derived words to their root. Lemmatization is the process of reducing a word to its lemma. The main difference between both methods is that lemmatization 	provides existing words, whereas stemming provides the root, which may not be an existing word
Stop words: words such as “what” or “the” .For this reason, they may represent noise that can be eliminated. 

Feature 	store
After scraping data and some cleaning process, it will be saved in some data store like MongoDB i.e. Feature store. A feature store is a centralised repository where you standardise the definition, storage, and access of features for training and serving. A feature store needs to provide real-time serving for the feature values, and to support both training and serving workloads.
With respect to this project we need to separate out labelled and unlabelled data in data store.

ML 	Pipeline

Data 				Preparation: The 				data is prepared for the ML task. We 				can use circular buffer to store data. The 				benefit of a circular buffer is, that you don't need 				infinite amounts of memory, since older entries get overridden 				automatically.This 				preparation involves data cleaning, where we will split the data 				into training, validation, and test sets. Also apply data 				transformations and feature engineering to the model that solves 				the target task. The output of this step are the data 				splits in 				the prepared format.

Model 				training: The different algorithms with the prepared data to 				train various ML models. We 				want to build neural network architecture which is expected to be 				able to classify news topics based on its content. We will be 				building LSTM model, as it works well with sequential, text data.

Model 				evaluation: The output of this step is a set of metrics to assess 				the quality of the model.

Model 				validation: The model is confirmed to be adequate for 				deployment—that its predictive performance is better than a 				certain baseline.
The output of ML pipeline is Trained model.

Model 	deployment
The validated model is deployed to a target environment to serve predictions. This deployment can be one of the following:
Microservices 	with a REST API to serve online predictions.
Part 	of a batch prediction system.

Trigger
To automate the ML production pipelines to retrain the models with new data, depending on the requirement:
Here we want to retrain ML model on a schedule: New, labelled data is systematically available for the ML system on a daily, weekly, or monthly basis. The retraining frequency also depends on how frequently the data patterns change, and how expensive it is to retrain your models.

# Solution Approach: 

For this project I need to web scrape a bunch of news articles online with 'BeautifulSoup', but I need to understand the unique html structures of each online news platform. Here, I did find a simple solution to my problems. I found Newspaper3k!

newspaper3k  allows us to find, extract, download, and parse articles from a potentially unlimited number of sources. Also, this works relatively seamlessly across multiple languages also. If that weren’t enough, newspaper is capable of caching articles, multi-threaded downloads, and basic NLP.

Yesturday, I encountered ‘Topic Modelling’ to classify documents (unstructured data) into their category in one of the blog. Latent Dirichlet Allocation (LDA) algorithm is one of the best approach to classify unstructred news.
So, after collecting news article's text/description I am trying with LDA algorithm.

LDA-NMF Combination Model

Latent Dirichlet Allocation (LDA) is a classic solution to Topic-Modelling. But in practice, it gave huge proportion of wrong classifications. Hence, Non Negative Matrix Factorization (NMF) is also used and numerically combined with LDA, along with Multi Class Binarizer to refine the results.

news_category.csv file has been used to save results, news text/description with dominant category.
