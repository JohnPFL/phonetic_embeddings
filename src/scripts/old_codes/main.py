from phonemizer import phonemize
from collections import Counter
import string
from nltk import word_tokenize
import nltk
nltk.download('toolbox')
from nltk.corpus import cmudict
from nltk.corpus import toolbox
import numpy as np
from scipy.spatial.distance import cosine

def get_most_similar_words(target_word, embeddings_dict, top_n=3):
    if target_word not in embeddings_dict:
        return None

    target_embedding = embeddings_dict[target_word]

    similarities = {}
    for word, embedding in embeddings_dict.items():
        if word != target_word:
            cosine_distance = cosine(target_embedding, embedding)
            similarities[word] = cosine_distance

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1])

    most_similar_words = sorted_similarities[:top_n]

    return most_similar_words


# Create a function to get the phonetic transcription of a word
def get_phonetic_transcription(word):
    entries = cmudict.entries()
    for entry in entries:
        if entry[0] == word:
            return entry[1]
    return None

def remove_non_alphanumeric(text):
    translator = str.maketrans('', '', string.punctuation.replace(' ', ''))
    return text.translate(translator)

def edit_distance(str1, str2):

    # Not considering the first and last letters, very important to cognitively retain a word
    if len(str1) > 5:
        str1 = str1[1:-1]
        str2 = str2[1:-1]

    m = len(str1)
    n = len(str2)
    
    # Create a 2D matrix to store the edit distances
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize the first row and column of the matrix
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Compute the edit distances
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    
    # Return the final edit distance
    return dp[m][n]


file_path = '/home/sperduti/sound_embeddings/Datasets/hate/train.txt'

with open(file_path, 'r') as file:
    lines = file.readlines()

data = []
texts = []

for line in lines:
    label, text = line.strip().split('\t')
    data.append((int(label), text))
    texts.append(text)

tokenized_texts  = [remove_non_alphanumeric(text).lower().split() for text in texts]
# Collect unique words
word_counts = Counter([word for text in tokenized_texts for word in text])
vocabulary = list(word_counts.keys())


# Convert the texts into phonetic representations using IPA
phonetic_texts = [phonemize(v, backend="espeak", language="en-us", strip=True) for v in vocabulary[:10]]


v_to_p = dict()

for v, p in zip(vocabulary, phonetic_texts):
    v_to_p[v] = p


# Calculate the edit distance between each pair of items
edit_distances = {}
keys = list(v_to_p.keys())
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        key1 = keys[i]
        key2 = keys[j]
        distance = edit_distance(v_to_p[key1], v_to_p[key2])
        edit_distances[(key1, key2)] = distance

# Print the edit distances
# A dictionary of correspondences with edit distance < 2 
corr_dict = {}
for (key1, key2), distance in edit_distances.items():
    if distance < 1:
        print(f"Edit distance between '{key1}' and '{key2}': {distance}")
        corr_dict[key1]= key2

print('ciao')
