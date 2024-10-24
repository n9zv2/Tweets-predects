import flask
import pickle
import joblib
import nltk
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
nltk.download('stopwords')

port_stem=PorterStemmer()
with open(f'C:/Users/khali/Desktop/ML-D/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

vectorizer = joblib.load('tfidf_vectorizer.pkl')


def stemming(content):
  stemmed_content=re.sub('[^a-zA-Z]',' ',content)
  stemmed_content=stemmed_content.lower()
  stemmed_content=stemmed_content.split()
  stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content=' '.join(stemmed_content)
  return stemmed_content



app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    if flask.request.method == 'POST':
        tweet = flask.request.form['tweet']
        df = pd.DataFrame([tweet], columns=['tweet'])
        df['tweet'] = df['tweet'].apply(stemming)
        final_text = df['tweet']
        final_text.iloc[0] = ''.join(final_text.iloc[0])
        final_text = vectorizer.transform(final_text)
        prediction = model.predict(final_text)

        return flask.render_template('index.html', result=prediction, original_input={"Tweet Analysis": tweet})
    
    # This handles GET requests (e.g., when the form is first loaded).
    return flask.render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)





#C:/Users/khali/Desktop/ML-D/trained_model.pkl
#C:/Users/khali/Desktop/ML-D/trained_model.pkl














# from flask import Flask, render_template, request
# import pickle

# # Use pickle to load in the pre-trained model.
# with open(f'C:/Users/khali/Desktop/ML-D/trained_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# with open(f'C:/Users/khali/Desktop/ML-D/trained_model.pkl', 'rb') as f:
#     vectorizer = pickle.load(f)

# app = Flask(__name__, template_folder='templates')

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         tweet = request.form['tweet']
#         # You can add code to process the tweet here, e.g., predict with the model
#     return render_template('index.html')

# if __name__ == "__main__":
#     app.run(debug=True)
