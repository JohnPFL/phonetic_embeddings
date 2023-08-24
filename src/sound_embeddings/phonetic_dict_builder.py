from phonemizer import phonemize
import string
from collections import Counter
import pickle

class PhoneticDictionaryBuilder:
    """
    PhoneticDictionaryBuilder is a class designed to process textual data,
    build a phonetic dictionary, and optionally save it to a file using pickle.

    This class offers methods for reading and preprocessing a dataset, generating
    a phonetic dictionary by converting words to their phonetic representations,
    and optionally saving the dictionary for later use.

    Attributes:
        dataset_path (str): Path to the dataset file containing labeled textual data.
        max_words (int): Maximum number of words to process from the dataset.

    Methods:
        read_dataset(): Read and process the dataset to extract textual data.
        remove_non_alphanumeric(text: str) -> str: Remove non-alphanumeric characters from text.
        phonemize(): Tokenize and convert dataset words to phonetic representations.
        build_phonetic_dictionary(dict_path: str = None) -> dict: Build and optionally save the phonetic dictionary.
    """
    def __init__(self, dataset_path, max_words:int):
        """
        Initialize the PhoneticDictionaryBuilder.

        Args:
            dataset_path (str): Path to the dataset file.
            max_words (int): Maximum number of words to process from the dataset.
        """
        self.dataset_path = dataset_path
        self.max_words = max_words
        self.phonetic_dict = {}

    def read_dataset(self):
        with open(self.dataset_path, 'r') as file:
            lines = file.readlines()
        texts = []
        for line in lines:
            label, text = line.strip().split('\t')
            texts.append(text)
        return texts

    def remove_non_alphanumeric(self, text):
        translator = str.maketrans('', '', string.punctuation.replace(' ', ''))
        return text.translate(translator)
    
    def phonemize(self):
        texts = self.read_dataset()
        tokenized_texts  = [self.remove_non_alphanumeric(text).lower().split() for text in texts]
        # Collect unique words
        word_counts = Counter([word for text in tokenized_texts for word in text])
        vocabulary = list(word_counts.keys())
        # Convert the texts into phonetic representations using IPA
        phonetic_texts = [phonemize(v, backend="espeak", language="en-us", strip=True) for v in vocabulary[:self.max_words]]
        return vocabulary, phonetic_texts

    def build_phonetic_dictionary(self, dict_path=None, csv_check=None):
        """
        Build the phonetic dictionary and optionally save it to a file using pickle.

        Args:
            dict_path (str, optional): Path to save the phonetic dictionary as a pickle file.

        Returns:
            dict: Phonetic dictionary with words as keys and phonetic representations as values.
        """
        vocabulary, phonetic_texts = self.phonemize()
        v_to_p = dict()
        for v, p in zip(vocabulary, phonetic_texts):
            v_to_p[v] = p

        if dict_path is not None: 
            # Save the dictionary to a file using pickle
            with open(dict_path, 'wb') as pickle_file:
                pickle.dump(v_to_p, pickle_file)

        return v_to_p
    
    @staticmethod
    def load_dictionary_from_pickle(dict_path):
        """
        Load a dictionary from a pickle file.

        Args:
            dict_path (str): Path to the pickle file containing the dictionary.

        Returns:
            dict: Loaded dictionary.
        """
        with open(dict_path, 'rb') as pickle_file:
            loaded_dict = pickle.load(pickle_file)
        return loaded_dict
    
def main():
    dataset_path = '/home/sperduti/sound_embeddings/Datasets/hate/train.txt'
    phonetic_dict = PhoneticDictionaryBuilder(dataset_path, 10)
    pd = phonetic_dict.build_phonetic_dictionary(dict_path='/home/sperduti/sound_embeddings/phonetic_dicts/hate_phonetic_embs.pickle')

if __name__ == '__main__':
    main()