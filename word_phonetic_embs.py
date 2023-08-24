############# TESTING WITH  PHONETIC WORD2VEC -- PROBABLY A DISASTER

tokenized_texts  = [remove_non_alphanumeric(text).lower().split() for text in texts]
# Collect unique words
word_counts = Counter([word for text in tokenized_texts for word in text])
vocabulary = list(word_counts.keys())

# Create a list to store phonetic transcriptions of the words
phonetic_transcriptions = []

# Iterate through the words and obtain their phonetic transcriptions
for word in vocabulary[:10]:
    phonetic_transcription = get_phonetic_transcription(word)
    if phonetic_transcription:
        phonetic_transcriptions.append(phonetic_transcription)

# Train Word2Vec model on phonetic transcriptions
model = Word2Vec(sentences=phonetic_transcriptions, vector_size=100, window=5, min_count=1, workers=4)


# Get phonetic embeddings for words
word_embeddings = {}
for word in vocabulary:
    phonetic_transcription = get_phonetic_transcription(word)
    if phonetic_transcription:
        embedding = model.wv[phonetic_transcription].mean(axis=0)
        word_embeddings[word] = embedding


# Print the word embeddings
for word, embedding in word_embeddings.items():
    print(f"{word}: {embedding}")

import pickle
with open("word_embs.pkl", "wb") as file:
    pickle.dump(word_embeddings, file)
#### FROM HERE ON OLD CODE 
