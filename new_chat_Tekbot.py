# Import and Load Dependencies

import json
import numpy as np
import random
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import os

nlp = spacy.load("en_core_web_sm")

# Load intents data
current_dir = os.path.dirname(os.path.abspath(__file__))
intents_path = os.path.join(current_dir, 'data', 'intents.json')
with open(intents_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

intents = data['intents']

# Prepare the data (pattern and tags)
all_patterns = []
all_tags = []
titles = [
    "asst",
    "asoc",
    "dr",
    "ms",
    "mr",
    "mrs",
    "prof",
    "engr",
    "ar"
]

def preprocess_text(sents):
    doc = nlp(sents)
    tokens = " ".join(set(token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.lemma_ not in titles))
    if len(tokens) == 0:
        return " ".join(set(token.lemma_.lower() for token in doc if not token.is_punct and token.lemma_ not in titles))
    else: return tokens

for intent in intents:
    for sents in intent['patterns']:
        tokens = preprocess_text(sents)
        if tokens not in all_patterns:
            all_patterns.append(tokens)
            all_tags.append(intent['tag'])

# Bag of Words with bigrams
vectorizer = CountVectorizer()
pattern_vectorizer = vectorizer.fit_transform(all_patterns)

def jaccard_similarity(user):
    """
    This function calculates the Jaccard similarity between the user's input and the patterns in the intents.
    
    Parameters:
    user (str): The user's input.
    
    Returns:
    max_similarity (float): The maximum Jaccard similarity found.
    max_similarity_index (int): The index of the pattern with the maximum Jaccard similarity.
    """
    
    user_ = preprocess_text(user)
    user_vector = vectorizer.transform([user_])

    similarities = []
    
    # Iterate the Vectorized Patterns
    for pattern_vector in pattern_vectorizer:
        # Calculate the Jaccard Similarity of the vectorized patterns
        intersection = np.minimum(user_vector.toarray(), pattern_vector.toarray()).sum()
        union = np.maximum(user_vector.toarray(), pattern_vector.toarray()).sum()
        # print(intersection, " | ", union)
        jaccard_sim = intersection / union if union != 0 else 0
        similarities.append(jaccard_sim)
        # print(user, " <-> ", pattern_vector, jaccard_sim)

    # Return the Index of the highest similarity
    max_similarity = max(similarities)
    max_similarity_index = similarities.index(max_similarity)
    return max_similarity, max_similarity_index

def get_response(max_similarity, index):
    """
    This function retrieves the appropriate response based on the maximum similarity found.
    
    Parameters:
    max_similarity (float): The maximum Jaccard similarity between the user's input and the patterns.
    index (int): The index of the pattern with the maximum Jaccard similarity.
    
    Returns:
    None. However, it prints the tag and the corresponding response if the maximum similarity is not zero.
    If the maximum similarity is zero, it prints a message indicating that the user's input was not understood.
    """
    response = ""
    context = ""
    # Fallback
    if max_similarity < 0.70:
        # print("Im Sorry, i did not understand what you said")
        response = "Im Sorry, i did not understand what you said"
        return [response, context]
    
    # Prints the corresponding response of the tag
    for intent in intents:
        if all_tags[index] == intent['tag']:
            # print("Tag: ", all_tags[index])
            # print("Response: ", intent['responses'][0])
            response = random.choice(intent['responses'])
            context = intent['context_set']
            return [response, context]

def get_Chat_response(text):
    max_similarity, max_similarity_index = jaccard_similarity(text)
    return get_response(max_similarity, max_similarity_index)