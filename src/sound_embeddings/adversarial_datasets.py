# Import necessary libraries and functions
import os

from sklearn.linear_model import LogisticRegression as lr
from sklearn.feature_extraction.text import TfidfVectorizer
# from lime.lime_text import LimeTextExplainer
import sys
from os.path import join
sys.path.insert(1, 'src/')

from sound_embeddings.phonetic_labeling import PhoneticLabeling
from phonemizer import phonemize


TECHNIQUES = ['RegressorClassifier']

def export(x, y, outpath):
        with open(outpath, 'wt') as foo:
            for x, y in zip(x, y):
                x = x.strip()
                foo.write(f'{y}\t{x}\n')

class RegressorClassifierMisspeller:
    """
    A class for building a misspelled word classifier using logistic regression.

    This class utilizes TF-IDF vectorization and logistic regression to select
    misspelled words based on training data.

    Args:
        top_k (int): Number of top features to consider for importance analysis.
        probability (float): Probability threshold for classification.

    Attributes:
        top_k (int): Number of top features considered for importance analysis.
        vectorizer (TfidfVectorizer): TF-IDF vectorizer for text data.
        lrc (LogisticRegression): Logistic regression classifier.
        probability (float): Probability threshold for classification.
        pos (list): Top positive feature names based on importance.
        neg (list): Top negative feature names based on importance.
    """
    def __init__(self, top_k, probability):
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(lowercase=False)
        self.lrc = lr(random_state=42, class_weight='balanced')
        self.probability = probability

    def f_importances(self):
        """
        Calculate and return top positive and negative feature names based on importance.

        Returns:
            tuple: A tuple containing lists of top positive and negative feature names.
        """
        imp = self.lrc.coef_[0]
        names = list(self.vectorizer.get_feature_names_out())
        num_features = len(imp)
        if num_features <= 2 * self.top_k:
            raise ValueError(f'There are less than {2*self.top_k} features!')
        
        sorted_importance, sorted_names = zip(*sorted(zip(imp, names), reverse=True))
        imp_pos = sorted_importance[:self.top_k]
        names_pos = sorted_names[:self.top_k]
        imp_neg = sorted_importance[-self.top_k:]
        names_neg = sorted_names[-self.top_k:]
        return names_pos, names_neg

    def fit(self, x, y):
        """
        Fit the model to the training data.

        Args:
            x (list or array-like): Input data.
            y (list or array-like): Target labels.

        Returns:
            None
        """
        x_transformed = self.vectorizer.fit_transform(x)
        self.lrc.fit(x_transformed, y)
        self.pos, self.neg = self.f_importances()


if __name__=='__main__':
    from data.data import TextDatasetHate

    datasetname = 'hs'
    hardness = 2000
    probability = 0.9
    technique = 'RegressorClassifier'
    misspelling_chars_path = './Datasets/misspellings.txt'
    label_csv_file_path = '/home/sperduti/sound_embeddings/Datasets/hate/phonetic_dict.csv'
    outpath = f'./Datasets/misspelled/{datasetname}/{technique}/{hardness}-{probability}'

    dataset = TextDatasetHate('/home/sperduti/sound_embeddings/Datasets/hate/phonemized_dataset')
    text = dataset.train_X + dataset.test_X
    y = list(dataset.train_y) + list(dataset.test_y)
    feature_importances = RegressorClassifierMisspeller(100, 0.8)
    important_features = feature_importances.fit(text, y)
    important_features = feature_importances.f_importances()
    # selecting only the most important words for class 1 in this case
    important_features = list(important_features[0])

    phonetic_data = {}
    phonetic_transcriptions = [phonemize(feature) for feature in important_features]
    for feature in range(len(important_features)):
        phonetic_data[important_features[feature]] = phonetic_transcriptions[feature]

    dataset_path = '/home/sperduti/sound_embeddings/Datasets/hate/train.txt'
    pdb = PhoneticLabeling(label_csv_file_path, phonetic_data)
    pdb.label_data()

    os.makedirs(outpath, exist_ok=True)