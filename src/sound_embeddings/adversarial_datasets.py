# Import necessary libraries and functions
import os

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression as lr
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
# from lime.lime_text import LimeTextExplainer
from sklearn.metrics import accuracy_score, f1_score, precision_score
import string
import random
import sys
from os.path import join
sys.path.insert(1, 'src/')
import data
import pathlib


# A misspelling dictionary. This misspelling dictionary is just a proof of concept. If it works, the idea should also work for
# more complex and not pre-designed symbols. 
# __misspell_dict = {'Q':'O',
#                 '0':'O',
#                 'O':['0', 'Q'],
#                 'W':'V',
#                 'w':'v',
#                 'V':'W',
#                 'v':['w','u', 'V'],
#                 'E':['F', '3'],
#                 'e':['é', 'è'],
#                 'é':['è', 'e'],
#                 'è':['e', 'é'],
#                 'B':'8',
#                 '8':'B',
#                 'g':['9', '8'],
#                 '9':'g',
#                 'z':['ʐ', 'ż', '2'],
#                 'ż':['z','ʐ'],
#                 'u':['υ', 'ս', 'ù', 'ü', 'ú', 'v'],
#                 'q':['զ', 'o'],
#                 'ù':'u',
#                  'Y':'V',
#                  'c':['C', 'ç'],
#                  'C':'c',
#                  '3':'E',
#                  'I':'l',
#                  'l':'I',
#                  '2':['Z', 'z'],
#                  's':['$','S','§'],
#                  '$':['s', 'S', '§'],
#                  'S':['$', '§'],
#                  '8':['B', 'g'],
#                  'B':'8',
#                  'G':'C',
#                  'j':'i',
#                  'i':['j', '!', '¡', 'l'],
#                  '!':'i',
#                  '¡':'i',
#                  'l':'i',
#                  'T':'I',
#                  'o':'ò',
#                  'ò':'o',
#                  'h':'n',
#                  'n':'h',
#                  'a':['à', 'ạ', 'ä', 'á'],
#                  'à':['a', 'ạ', 'ä', 'á'],
#                  'ç':'c',
#                  '4':'A',
#                  'A':'4',
#                  'Z':'7',
#                  '7':'Z',
#                  '§':['S', 's'],
#                 }


TECHNIQUES = ['RegressorClassifier']

def export(x, y, outpath):
        with open(outpath, 'wt') as foo:
            for x, y in zip(x, y):
                x = x.strip()
                foo.write(f'{y}\t{x}\n')

class SoundAdversarialCorruption:
    def __init__(self, dataset, probability):
        self.dataset = dataset
        self.probability = probability

class RegressorClassifierMisspeller(SoundAdversarialCorruption):
    def __init__(self, top_k, probability):
        super().__init__(top_k, probability)
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(lowercase=False)

    def f_importances(self):
        imp = self.lrc.coef_[0]
        names = list(self.vectorizer.get_feature_names_out())
        assert len(imp) > 2*self.top_k, f'there are less than {2*self.top_k} features!'
        # pos
        sorted_importance, sorted_names = zip(*sorted(zip(imp, names), reverse=True))
        imp_pos = sorted_importance[:self.top_k]
        names_pos = sorted_names[:self.top_k]
        # neg
        imp_neg = sorted_importance[-self.top_k:]
        names_neg = sorted_names[-self.top_k:]
        return names_pos, names_neg

    def fit(self, x, y):
        self.lrc = lr(random_state=42, class_weight='balanced')
        self.lrc.fit(self.vectorizer.fit_transform(x), y)
        self.pos, self.neg = self.f_importances()


if __name__=='__main__':
    from data.data import TextDatasetHate

    datasetname = 'hs'
    hardness = 2000
    probability = 0.9
    technique = 'RegressorClassifier'
    misspelling_chars_path = './Datasets/misspellings.txt'
    outpath = f'./Datasets/misspelled/{datasetname}/{technique}/{hardness}-{probability}'

    dataset = TextDatasetHate()
    text = dataset.train_X + dataset.test_X
    y = list(dataset.train_y) + list(dataset.test_y)
    feature_importances = RegressorClassifierMisspeller(100, 0.8)
    test = feature_importances.fit(text, y)
    test = feature_importances.f_importances()

    os.makedirs(outpath, exist_ok=True)


    

    #mock - train
    #new_dataset = RegressorClassifierMisspeller(2000, dataset.train_X, dataset.train_y, __misspell_dict, cle_path, probability).generate_mock()
    #RegressorClassifierMisspeller.export(join(outpath, 'train.txt'), new_dataset)

    #mock - test
    #new_dataset = RegressorClassifierMisspeller(2000, dataset.test_X, dataset.test_y, __misspell_dict, cle_path, probability).generate_mock()
    #RegressorClassifierMisspeller.export(join(outpath, 'test.txt'), new_dataset)

   