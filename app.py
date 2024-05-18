from flask import Flask, render_template, request, jsonify

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
from sklearn.naive_bayes import MultinomialNB
import joblib
import nltk
import random
import json
import pickle
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')
    # return "Hello Evy!"

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

def load_model():
    # Load the model from file
    model = joblib.load("./model/naive_bayes_model.pkl")
    vectorizer = joblib.load("./model/vectorizer.pkl")
    with open("./data/intents.json", 'r', encoding='utf-8') as file:
        data = json.load(file)

    return model, vectorizer, data

def preprocess_input(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if len(token) > 1]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def get_Chat_response(text):    
    naive_bayes_model, vectorizer, data = load_model()
    
    input_text = preprocess_input(text)
    input_text = vectorizer.transform([input_text])
    
    predicted_intent = naive_bayes_model.predict(input_text)[0]
    
    for intent in data['intents']:
        if intent['tag'] == predicted_intent:
            response = random.choice(intent['responses'])
            context = intent['context_set']
            break
    print('Response: ', response)
    return [response, context]

if __name__ == '__main__':
    app.run(debug=True)