import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

pickle.dump(Pipeline([('vectorizer', TfidfVectorizer()), ('classifier',LinearRegression())]), open('model.pkl', 'wb'))