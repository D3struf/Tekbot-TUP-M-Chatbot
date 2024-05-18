import nltk
from nltk.stem import WordNetLemmatizer
import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load intents data
with open("./data/intents.json", 'r', encoding='utf-8') as file:
    data = json.load(file)

intents = data['intents']

# Prepare training data
patterns = []
tags = []

for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Vectorize patterns
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

def get_response(user_input):
    user_input_vector = vectorizer.transform([user_input])
    cosine_similarities = linear_kernel(user_input_vector, X).flatten()
    most_similar_index = np.argmax(cosine_similarities)
    predicted_tag = tags[most_similar_index]

    for intent in intents:
        if intent['tag'] == predicted_tag:
            response = random.choice(intent['responses'])
            context = intent['context_set']
    
    return [response, context]

lemmatizer = WordNetLemmatizer()

def preprocess_input(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if len(token) > 1]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def get_Chat_response(text):
    input_text = preprocess_input(text)
    return get_response(input_text)