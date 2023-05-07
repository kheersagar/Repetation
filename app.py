import nltk
import requests
import re
try:
    from nltk.corpus import stopwords
except:
    nltk.download('stopwords')
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
df = pd.read_csv(r'Aptitude.csv')
Id = list(range(0, 2025))
df['Id'] = Id

@app.route('/calculate_score', methods=['POST'])
def calculate_score():
    data = request.get_json()

    level = data['level']
    is_correct = data['is_correct']
    point = data['point']
    
    # Calculate points for the question
    if len(point) == 1  and -1 in point:
        points = []
    else:
        points = [*point]

    value = find_values_of_x(level, is_correct)
    points.append(value)

    # Calculate interval score
    interval_score = find_interval(points)

    # Return interval score as JSON
    return jsonify({'point': points,'interval_score': interval_score})

def find_values_of_x(level, is_correct):
    if is_correct == 'wrong' and level == 'H':
        value = 0
    elif is_correct == 'wrong' and level == 'M':
        value = 1
    elif is_correct == 'wrong' and level == 'E':
        value = 2
    elif is_correct == 'right' and level == 'H':
        value = 3
    elif is_correct == 'right' and level == 'M':
        value = 4
    elif is_correct == 'right' and level == 'E':
        value = 5
    else:
        value = 0

    return value

def find_interval(points, a=6.0, b=-0.8, c=0.28, d=0.02, theta=0.07) -> float:
    assert all(0 <= point_i <= 5 for point_i in points)
    correct_points = [point_i >= 3 for point_i in points]
    
    # If you got the last question incorrect, just return 1
    if not correct_points[-1]:
        return 1.0
    
    # Calculate the latest consecutive answer streak
    num_consecutively_correct = 0   
    for correct in reversed(correct_points):
        if correct:
            num_consecutively_correct += 1
        else:
            break
    return round(a * (max(1.3, 2.5 + sum(b+c*point_i+d*point_i*point_i for point_i in points)))**(theta*num_consecutively_correct))



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
