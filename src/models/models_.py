
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.svm import LinearSVC # type: ignore
from module.architectures import CNNbase, PhoneticCNN
from utils.util import CharVocabulary, PhoneticCharVocabulary
import torch
import sys
from os.path import join
from models.classification_functions import train_model, predict_samples


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

    def __init__(self, batch_size, learning_rate, seed, 
                 model_name, patience=0, nepochs=500, device='cuda', paths = {'tmp_path': 'tmp', 
                                                       'def_path': 'def'}, **kwargs):
        self.batch_size = batch_size
        self.nepochs = nepochs
        self.device = device
        self.model_name = model_name
        self.patience = patience
        self.name = 'cnn'
        self.learning_rate = learning_rate
        # seed for the validation split
        self.seed = seed
        self.paths = paths
        self.kwargs = kwargs

        
    def fit(self, X, y, datasetname):
        self.char_encoder = CharVocabulary(X)
        vocab_length = len(self.char_encoder)+1
        self.model = CNNbase(vocab_length, **self.kwargs).to(self.device)
        self.latest_model = train_model(self.model, X, y, self.char_encoder, datasetname,
                                        self.batch_size, self.seed, nepochs=self.nepochs, 
                                        model_name=self.model_name, 
                                        patience= self.patience, lazy_loader=True, 
                                        device=self.device, lr=self.learning_rate,
                                        tmp_path= self.paths['tmp_path'], def_path= self.paths['def_path'])

    def predict(self, X):
        if self.patience:
            print(f'latest best model is... {self.latest_model =}')
            model = torch.load(self.latest_model)
            return predict_samples(model, X, self.char_encoder, self.batch_size, lazy_loader=True, device=self.device)
        else:
            return predict_samples(self.model, X, self.char_encoder, self.batch_size, lazy_loader=True, device=self.device)


class CNNPhoneticClassifier(Model):

    def __init__(self, batch_size, phonetic_dataset, 
                 save_folder,
                  learning_rate, seed, 
                 model_name = 'False', 
                 patience = 0, nepochs = 500, paths = {'tmp_path': 'tmp', 
                                                       'def_path': 'def'}, **kwargs):
        self.batch_size = batch_size
        self.nepochs = nepochs
        self.device = 'cuda'
        self.model_name = model_name
        self.patience = patience
        self.kwargs = kwargs
        self.name = 'phonetic_cnn'
        self.learning_rate = learning_rate
        self.phonetic_dataset = phonetic_dataset
        self.save_folder = save_folder
        # seed for the validation split
        self.seed = seed 
        self.paths = paths

    def fit(self, X, y, datasetname):
        # Fit the classifier to the training data
        self.char_encoder = PhoneticCharVocabulary(X, self.phonetic_dataset, self.save_folder)
        vocab_length = len(self.char_encoder)+1
        self.model = PhoneticCNN(vocab_length, **self.kwargs).to('cuda')
        self.latest_model = train_model(self.model, X, y, self.char_encoder, 
                                        datasetname, self.batch_size, self.seed, 
                                        nepochs= self.nepochs,
                                        model_name=self.model_name, patience=self.patience, 
                                        lazy_loader=True, device=self.device, lr=self.learning_rate,
                                        tmp_path= self.paths['tmp_path'], def_path= self.paths['def_path'])
    
    def predict(self, X):
        if self.patience:
            print(f'latest best model is... {self.latest_model =}')
            model = torch.load(self.latest_model)
            return predict_samples(model, X, self.char_encoder, self.batch_size, lazy_loader=True, device=self.device)
        else:
            return predict_samples(self.model, X, self.char_encoder, self.batch_size, lazy_loader=True, device=self.device)
        