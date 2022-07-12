from flask import Flask, render_template, request, jsonify
app = Flask(__name__)
import praw
import sys
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
name = ""
@app.route("/")
def home():
    return render_template('index.html')
@app.route('/process', methods=['POST'])
def process():


    name = request.form['name']

    if name:
        df = get_hot_titles(name)
        df = get_title_length(df)
        title_length_graph(df)
        df = get_word_count(df)
   
        graph_scatter_plot(df, 'Word count', 'ups', "Word count", "Number of Upvotes","Effect of Word Count on Number of Upvotes ")
        df = unweighted_word_count(df)
        for i in df.columns[4:]:
            df[i] = df[i] * df['ups']
        df = df.append(df.sum().rename('Total'))      

        return jsonify({'name' : " ".join(df.iloc[-1][4:-1].astype(int).nlargest(25).keys().values.tolist()), 'val':' '.join(str(x) for x in df.iloc[-1][4:-1].astype(int).nlargest(25).values.tolist())})




@app.route('/visualize')
def visualize():
    name = request.form['name']
    subreddit = reddit.subreddit('python')
    hot = subreddit.hot(limit=20)
    df = get_hot_titles('python')

    model = pickle.load(open('model.pkl', 'rb'))
    model.fit(df.title,df.ups)
    model.predict(['python'])
    
    df = get_title_length(df)

    title_length_graph(df)
    df = get_word_count(df)
   
    graph_scatter_plot(df, 'Word count', 'ups', "Word count", "Number of Upvotes","Effect of Word Count on Number of Upvotes ")
    df = unweighted_word_count(df)
  
    plot_weighted_keywords(df)
    img = title_length_graph(df)
    weighted_keywords = plot_weighted_keywords(df)
    
    return send_file(img,mimetype='img/png')
@app.route('/weighted_keyword')
def weighted_keyword():
    name = request.form['name']
    subreddit = reddit.subreddit('python')
    hot = subreddit.hot(limit=20)
    df = get_hot_titles('python')

    model = pickle.load(open('model.pkl', 'rb'))
    model.fit(df.title,df.ups)
    model.predict(['python'])
    
    df = get_title_length(df)

    title_length_graph(df)
    df = get_word_count(df)
   
    graph_scatter_plot(df, 'Word count', 'ups', "Word count", "Number of Upvotes","Effect of Word Count on Number of Upvotes ")
    df = unweighted_word_count(df)
  
    plot_weighted_keywords(df)
    img = title_length_graph(df)
    weighted_keywords = plot_weighted_keywords(df)
    print(df.iloc[-1][4:-1].values.sort(), file=sys.stderr)
    return send_file(weighted_keywords,mimetype='img/png')