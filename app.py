from flask import Flask
app = Flask(__name__)
import praw
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from model import *
import pickle
nltk.download('punkt')
nltk.download('stopwords')
plt.switch_backend('Agg') 
@app.route("/")
def home():
    subreddit = reddit.subreddit('python')
    hot = subreddit.hot(limit=20)
    df = get_hot_titles()

    model = pickle.load(open('model.pkl', 'rb'))
    model.fit(df.title,df.ups)
    model.predict(['python'])
    
    df = get_title_length(df)

    title_length_graph(df)
    df = get_word_count(df)
   
    graph_scatter_plot(df, 'Word count', 'ups', "Word count", "Number of Upvotes","Effect of Word Count on Number of Upvotes ")
    df = unweighted_word_count(df)
  
    plot_weighted_keywords(df)
    s = ""
    s += df['title'][1]
    return str(model.predict(['python'])[0])