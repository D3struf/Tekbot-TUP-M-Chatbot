import nltk
import random
import json
import os
import pickle
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
nltk.download('wordnet')

# Load intents from JSON file
with open("./data/intents.json", 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Function to perform synonym replacement
def synonym_replacement(tokens, limit):
    augmented_sentences = []
    for i in range(len(tokens)):
        synonyms = []
        for syn in wordnet.synsets(tokens[i]):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        if len(synonyms) > 0:
            num_augmentations = min(limit, len(synonyms))
            sampled_synonyms = random.sample(synonyms, num_augmentations)
            for synonym in sampled_synonyms:
                augmented_tokens = tokens[:i] + [synonym] + tokens[i+1:]
                augmented_sentences.append(' '.join(augmented_tokens))
    return augmented_sentences

text_data = []
labels = []
stopwords = set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


limit_per_tag = 1000

for intent in intents['intents']:
    augmented_sentences_per_tag = 0
    for example in intent['patterns']:
        tokens = nltk.word_tokenize(example.lower())
        filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords and token.isalpha()]
        if filtered_tokens:
            text_data.append(' '.join(filtered_tokens))
            labels.append(intent['tag'])
            
            augmented_sentences = synonym_replacement(filtered_tokens, limit_per_tag - augmented_sentences_per_tag)
            for augmented_sentence in augmented_sentences:
                text_data.append(augmented_sentence)
                labels.append(intent['tag'])
                augmented_sentences_per_tag += 1
                if augmented_sentences_per_tag >= limit_per_tag:
                    break

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text_data)
y = labels

def naive_bayes_model(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=100)

    name, model, param_grid = ('Multinomial Naive Bayes', MultinomialNB(), {'alpha': [0.1, 0.3, 0.5, 0.8, 1.0]})

    grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f'{name}: {score:.4f} (best parameters: {grid.best_params_})')

    return grid

naive_model = naive_bayes_model(X, y)

if not os.path.exists('./model'):
    os.makedirs('model')

# Save the trained model
with open('./model/naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(naive_model, f)

# Save the vectorizer
with open('./model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)