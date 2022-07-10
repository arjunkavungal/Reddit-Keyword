import streamlit as st
import praw
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
reddit = praw.Reddit(client_id="p1lt136fs51SWOv6zlM6QA",client_secret="Ffth4WUUPhmFO4_b6oUdlmY5e3ZOYA",
                     username="Hot-Helicopter5986",password="",user_agent="a")
keyword = st.text_input("Enter the topic that you want to research keywords for")
if st.button('Say hello'):
    subreddit = reddit.subreddit(keyword)

    hot = subreddit.hot(limit=20)
    df = pd.DataFrame()
    for i in hot:
        df = df.append({'title': i.title,'ups':i.ups}, ignore_index=True)

    st.dataframe(df)
    tvec = TfidfVectorizer()
    clf = LinearRegression()

    model = Pipeline([('vectorizer', tvec), ('classifier',clf)])
    model.fit(df.title,df.ups)
    title = st.text_input("Put the keywords that you want to predict")
    st.write(model.predict(title))
    df['Title length'] = len(df['title'])
    for i in range(len(df)):
        df['Title length'][i] = len(df['title'][i])

    q_low = df["Title length"].quantile(0.01)
    q_hi  = df["Title length"].quantile(0.99)

    df_filtered = df[(df["Title length"] < q_hi) & (df["Title length"] > q_low)]
    plt.scatter(df_filtered['Title length'],df_filtered.ups)
    plt.show()
    a = []
    for i in range(len(df)):
        a.append(len(df['title'][i].split(' ')))
    df['Word count'] = a
    plt.scatter(df['Word count'],df.ups)
    plt.show()
    s = ""
    for i in range(len(df)):
        s += df['title'][i] + " "




    text_tokens = word_tokenize(s)

    tokens_without_sw = [word.lower() for word in text_tokens if not word.lower() in stopwords.words()]

    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in s:
        if ele in punc:
            s = s.replace(ele, "")

    s = ' '.join(tokens_without_sw)
    punc = '''!()–-[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in s:
        if ele in punc:
            s = s.replace(ele, "")
    df['title'] = df['title'].astype(str)
    df["count"]= df["title"].str.get(0)
    df = df.iloc[:20,:4]
    df['title'].map(lambda x: x.lower() if isinstance(x,str) else x)

    for j in range(len(df)):
        for i in s.split(' '):
            df.at[j,i] = df['title'][j].count(i)
    df = df.append(df.sum().rename('Total'))
    df.iloc[-1][4:-1].astype(int).nlargest(25).plot(kind="bar")
    plt.rcParams["figure.figsize"] = (6,6)
    plt.show()
    for i in df.columns[4:]:
        df[i] = df[i] * df['ups']
    df = df.append(df.sum().rename('Total'))
    df.iloc[-1][4:-1].astype(int).nlargest(25).plot(kind="bar")
    plt.rcParams["figure.figsize"] = (6,6)
    plt.show()