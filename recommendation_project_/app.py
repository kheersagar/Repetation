import nltk
import requests
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the data and preprocess it
df = pd.read_csv(r'C:\Users\sonali\Downloads\recommendation_project_2.0\recommendation_project_\Aptitude all set.csv')
Id = list(range(0, 85))
df['Id'] = Id

# Clean the data
def text_cleaner(text):
    clean_text = re.sub(r'@[A-Za-z0-9]+', '', text)
    clean_text = re.sub('#', '', clean_text)
    clean_text = re.sub(r"'s\b", '', clean_text)
    clean_text = re.sub(r'[%$#@&}{]', '', clean_text)
    clean_text = re.sub(r'[.,:;!]', '', clean_text)
    clean_text = re.sub(r"http\S+", "", clean_text)
    letters_only = re.sub("[^a-zA-Z]", ' ', clean_text)
        
    lower_case = letters_only.lower()
    tokens = [w for w in lower_case.split() if not w in stop_words]
    clean_text = ''
    for i in tokens:
        clean_text = clean_text + lemmatizer.lemmatize(i)+ ' '
    return clean_text.strip()

def clean():
    cleaned_text = []
    for i in df['question']:
        cleaned_text.append(text_cleaner(i))
    df['Questions_cleaned1'] = cleaned_text

clean()

# Vectorize the text
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(df['Questions_cleaned1'])
vectors_df = pd.DataFrame.sparse.from_spmatrix(vectors)

# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(vectors)

@app.route('/recommend', methods=['POST'])
def recommend():
    # Parse the input
    data = request.json
    question = data.get('question')
    n = data.get('n', 5)
    
    if not question:
        return 'Error: No question provided'
    
    # Find the most similar questions to the input question
    question_cleaned = text_cleaner(question)
    question_vector = vectorizer.transform([question_cleaned])
    similarities = cosine_similarity(question_vector, vectors)[0]
    similar_indices = similarities.argsort()[::-1][:n]
    recommended_questions = df.iloc[similar_indices][['Id', 'question']].to_dict('records')
    
    # Return the recommendations
    return jsonify(recommended_questions)

if __name__ == '__main__':
    app.run(debug=True)
