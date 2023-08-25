import sys
sys.path.append('/home/sperduti/sound_embeddings/src')
from data.data import TextDatasetHate
from sound_embeddings.misspeller import PhoneticMisspeller
from os.path import join
import numpy as np

phonetic_equivalences = '/home/sperduti/sound_embeddings/sound_embeddings_utils/phonetic_subwords_dataset/hate/phonetic_dict.csv'
misspelled_correspondences_path = '/home/sperduti/sound_embeddings/Datasets/hate/misspelled_datasets/hate'
misspelled_dataset_path = '/home/sperduti/sound_embeddings/Datasets/hate/misspelled_datasets/hate'

data = TextDatasetHate()

misspeller = PhoneticMisspeller(phonetic_equivalences)

train_X = data.train_X
train_y = data.train_y

test_x = data.test_X[:10]
test_y = data.test_y[:10]

#apply the misspellings 
misspelled_test_x = misspeller.batch_misspell_dataset_parallel(test_x, test_y, misspelled_dataset_path, batch_size=4)

print('ciao')

# (2) Qui va inserita "generate and print datasets" e prima il check sull'esistenza del dataset

# (3) 



