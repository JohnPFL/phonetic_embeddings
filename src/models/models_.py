from abc import ABC, abstractmethod
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics import f1_score, recall_score, precision_score # type: ignore
from sklearn.model_selection import train_test_split # type: ignore 
from sklearn.svm import LinearSVC # type: ignore
from torch.utils.data import DataLoader
from data.data import TextDataset, NeuralDataset
from data import data 
from module.architectures import CNNbase
from utils.util import CharVocabulary
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.util import EarlyStop
import pandas as pd
import os
import sys
import shutil
import time
from os.path import join
import torch.nn.functional as F
import torch.nn.utils as utils
from classification_functions import train_model, predict_samples


class Model(ABC):

    @abstractmethod
    def fit(self, X, y):
        ...

    @abstractmethod
    def predict(self, X):
        ...

# Definition of a Linear SVM
class SVMclassifier(Model):

    def __init__(self, **kwargs):
        self.name = 'svm'
        self.vectorizer = TfidfVectorizer(**kwargs)
        self.svm = LinearSVC()

    def fit(self, X, y):
        X = self.vectorizer.fit_transform(X)
        self.svm.fit(X,y)
        return self

    def predict(self, X):
        X = self.vectorizer.transform(X)
        return self.svm.predict(X)
    
class CNNbaseClassifier(Model):

    def __init__(self, batch_size, learning_rate, seed, model_name, patience=False, nepochs=500, device='cuda', **kwargs):
        self.batch_size = batch_size
        self.nepochs = nepochs
        self.device = device
        self.model_name = model_name
        self.patience = patience
        self.kwargs = kwargs
        self.name = 'cnn'
        self.learning_rate = learning_rate
        # seed for the validation split
        self.seed = seed
        
    def fit(self, X, y, datasetname):
        self.char_encoder = CharVocabulary(X)
        vocab_length = len(self.char_encoder)+1
        self.model = CNNbase(vocab_length, **self.kwargs).to(self.device).xavier_uniform()
        self.latest_model = train_model(self.model, X, y, self.char_encoder, datasetname,
                                        self.batch_size, self.seed, nepochs=self.nepochs, model_name=self.model_name, 
                                        patience= self.patience, lazy_loader=True, device=self.device, lr=self.learning_rate)

    def predict(self, X):
        if self.patience:
            print(f'latest best model is... {self.latest_model =}')
            model = torch.load(self.latest_model)
            return predict_samples(model, X, self.char_encoder, self.batch_size, lazy_loader=True, device=self.device)
        else:
            return predict_samples(self.model, X, self.char_encoder, self.batch_size, lazy_loader=True, device=self.device)
