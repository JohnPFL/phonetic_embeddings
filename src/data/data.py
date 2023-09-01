from os.path import join
from functools import cached_property

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.util import Encoder

import string


DATASETS = ['hate']


class TextDataset:
    def __init__(self, path):
        self.train_X, self.train_y = TextDataset._read(join(path, 'train.txt'))
        self.test_X, self.test_y = TextDataset._read(path)
        self.classes = np.unique(self.train_y).sort()

    def __repr__(self):
        str = f'#training documents = {len(self.train_X)}\n'
        str += f'#test documents = {len(self.test_X)}'
        return str

    @cached_property
    def word_vocabulary(self):
        return TextDataset.vocabulary(self.train_X + self.test_X)

    @cached_property
    def train_vocabulary(self):
        return TextDataset.vocabulary(self.train_X)

    @cached_property
    def test_vocabulary(self):
        return TextDataset.vocabulary(self.test_X)

    @classmethod
    def vocabulary(cls, X):
        return sorted(list(set(' '.join(X))))

    @classmethod
    def _read(cls, path):
        with open(path, 'rt') as fin:
            y, X = zip(*[line.split('\t') for line in fin.readlines()])
            y = list(map(int, y))
            y = np.asarray(y, dtype=int)
            X = tuple(sentence.strip() for sentence in X)
            return X, y

class TextDatasetHate(TextDataset):
    def __init__(self, phonemized_path):
        super().__init__(f'/sound_embeddings/Datasets/hate')
        self.name = 'hate'
        self.train_X_phonetics, self.train_y = TextDataset._read(join(phonemized_path, 'train.txt'))
        self.test_x_phonetics, self.test_y = TextDataset._read(join(phonemized_path, 'test.txt'))
        self.test_x_misspelled_phonetics, self.test_y = TextDataset._read(join(phonemized_path, 'test_misspelled.txt'))


    def __repr__(self):
        return 'Dataset Hate\n' + super().__repr__()

class MultipathDataset(TextDataset):
    def __init__(self, clean_path, adv_path, phonetic = False, hard=False, adversarial = False):
        self.train_X, self.train_y = TextDataset._read(join(clean_path, 'train.txt'))
        self.test_X, self.test_y = TextDataset._read(join(adv_path, 'test.txt'))
        if phonetic:
            if adversarial:
                self.test_X, self.test_y = TextDataset._read(join(adv_path, 'test_misspelled.txt'))

        # Get unique classes and sort them
        self.classes = np.unique(self.train_y)
        self.classes.sort()
        
        self.name = 'adv'

        # Remove punctuations and special characters from train_X and test_X
        if phonetic:
            self.train_X = [self._remove_special_characters(text) for text in self.train_X]
            self.test_X = [self._remove_special_characters(text) for text in self.test_X]



    def _remove_special_characters(self, text):
        # Remove punctuations using string.punctuation
        translator = str.maketrans('', '', string.punctuation)
        text_without_punctuations = text.translate(translator)
        
        # Remove specific character 'ː'
        text_without_special_char = text_without_punctuations.replace('ː', '')
        
        return text_without_special_char

    def __repr__(self):
        return 'Dataset adversarial\n' + super().__repr__()

class NeuralDataset(Dataset):

    # It takes in input a training column and a label column
    def __init__(self, X, y, encoder: Encoder, MAX_LENGTH=2000, lazy_encoding=False, embedder_for_repr=False, kernel_size=7):
        self.encoder = encoder
        self.max_length = MAX_LENGTH
        self.post_pone_encoding = lazy_encoding
        self.embedder_for_repr = embedder_for_repr
        self.kernel_size = kernel_size
        self.instances = X if lazy_encoding else [encoder.encode(sentence[:MAX_LENGTH]) for sentence in X]
        self.labels = y

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        if self.post_pone_encoding:
            instance = self.encoder.encode(instance[:self.max_length])

        if self.labels is None:
            return instance
        else:
            label = self.labels[idx]
            return instance, label

    def asDataLoader(self, batch_size, shuffle=False):
        unk = len(self.encoder) # type: ignore
        # this case is useful when using the cnn representation layer before the classifier

        def pad(X, repr_cnn= self.embedder_for_repr, kernel_size = self.kernel_size):
            longest_sentence = max(map(len, X))
            # to avoid kernel error (kernel size cannot be bigger than input size)
            if longest_sentence < kernel_size:
                longest_sentence = kernel_size + 1
            if isinstance(X[0], list) and self.embedder_for_repr is False:
                Xpadded = [instance + [unk] * (longest_sentence - len(instance)) for instance in X]
                Xpadded = torch.tensor(Xpadded)
            else:
                Xpadded = np.zeros(shape=(len(X), longest_sentence, len(X[0][0])))
                for i, xi in enumerate(X):
                    Xpadded[i,:len(xi),:]=xi
                Xpadded = np.asarray(Xpadded)
                Xpadded = torch.tensor(Xpadded, dtype=torch.float32)
            return Xpadded

        def collate_fn(batch):
            X, y = zip(*batch)
            X = pad(X)
            y = torch.tensor(y, dtype=torch.float32)

            return X, y

        def collate_unsupervised_fn(batch):
            return pad(batch)

        collate = collate_fn
        if self.labels is None:
            collate = collate_unsupervised_fn

        return DataLoader(self, collate_fn=collate, shuffle=shuffle, batch_size=batch_size)



if __name__ == '__main__':
    inpath = '../Datasets/hate'
    data = TextDataset(inpath)
    print(data)

    data = TextDatasetHate('/sound_embeddings/Datasets/hate/phonemized_dataset')
    print(data)

    print(data)

