from abc import ABC, abstractmethod
import os 
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer # type: ignore
import sys
sys.path.append('/home/sperduti/sound_embeddings/src')
from sound_embeddings.misspeller import PhoneticMisspeller
from phonemizer import phonemize
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import resource


class Vocabulary:
    def __init__(self, elements):
        self.id2el = sorted(list(set(elements)))
        self.el2id = {el:id for id, el in enumerate(self.id2el)}

    def __len__(self):
        return len(self.id2el)
    
class Encoder(ABC):
    @abstractmethod
    def encode(self, instance):
        ...

class PhoneticVocabulary:
    def __init__(self, elements, phonetic_dataset, save_folder=None, use_existing=True):
        self.save_folder = save_folder
        self.use_existing = use_existing

        if self.use_existing and self.save_folder and self._check_existing_vocabulary():
            self._load_existing_vocabulary()
        else:
            self._train_vocabulary(elements, phonetic_dataset)

    def _train_vocabulary(self, elements, phonetic_dataset):
        phonetic_misspeller = PhoneticMisspeller(phonetic_dataset)
        phonetic_elements = phonetic_misspeller.batch_phonemize_parallel(elements[:10], "en-us")
        self.id2el = sorted(list(set(' '.join(phonetic_elements)))) # type: ignore
        self.el2id = {el: id for id, el in enumerate(self.id2el)}
        
        if self.save_folder:
            self._save_vocabulary()

    def _save_vocabulary(self):
        with open(os.path.join(self.save_folder, "vocabulary.pkl"), "wb") as f:  # type: ignore
            pickle.dump((self.id2el, self.el2id), f)

    def _load_existing_vocabulary(self):
        with open(os.path.join(self.save_folder, "vocabulary.pkl"), "rb") as f: # type: ignore
            self.id2el, self.el2id = pickle.load(f)
    
    def _check_existing_vocabulary(self):
        return os.path.exists(os.path.join(self.save_folder, "vocabulary.pkl")) # type: ignore
    
    def __len__(self):
        return len(self.id2el)


class CharVocabulary(Vocabulary, Encoder):
    def __init__(self, elements):
        super().__init__(' '.join(elements))

    def encode(self, instance):
        unk = len(self)
        return [self.el2id.get(el, unk) for el in list(instance)]

class PhoneticCharVocabulary(PhoneticVocabulary, Encoder):
    def __init__(self, elements, phonetic_dataset, save_folder=None, use_existing=True):
        super().__init__(' '.join(elements), phonetic_dataset, save_folder, use_existing)
        self.phonetic_dataset = phonetic_dataset

    def encode(self, instance):
        phonetic_misspeller = PhoneticMisspeller(self.phonetic_dataset)
        instance = phonetic_misspeller.batch_phonemize_parallel(instance, "en-us", verbose = False)
        instance = ' '.join(instance) # type: ignore
        unk = len(self)
        return [self.el2id.get(el, unk) for el in list(instance)]


class WordVocabulary(Vocabulary, Encoder):
    def __init__(self, elements):
        self.analyzer = CountVectorizer().build_analyzer()
        allwords = []
        for text in elements:
            words = self.analyzer(text)
            allwords.extend(words)
        super().__init__(allwords)

    def encode(self, instance):
        unk = len(self)
        return [self.el2id.get(el, unk) for el in self.analyzer(instance)]



class VisuallyGroundedVocabulary(Vocabulary, Encoder):
    def __init__(self, elements, char2visual_fn):
        super().__init__(' '.join(elements))
        self.char2visual_fn = char2visual_fn

    def encode(self, instance):
        return np.asarray([self.char2visual_fn(el) for el in list(instance)])
    
    def encode_for_repr(self, instance):
        pil_images = [self.char2visual_fn(image).copy for image in instance]
        return pil_images


class EarlyStop:
    """
    A class implementing the early-stopping condition typically used for training neural networks.

    >>> earlystop = EarlyStop(patience=2, lower_is_better=True)
    >>> earlystop(0.9, epoch=0)
    >>> earlystop(0.7, epoch=1)
    >>> earlystop.IMPROVED  # is True
    >>> earlystop(1.0, epoch=2)
    >>> earlystop.STOP  # is False (patience=1)
    >>> earlystop(1.0, epoch=3)
    >>> earlystop.STOP  # is True (patience=0)
    >>> earlystop.best_epoch  # is 1
    >>> earlystop.best_score  # is 0.7

    :param patience: the number of (consecutive) times that a monitored evaluation metric (typically obtaind in a
        held-out validation split) can be found to be worse than the best one obtained so far, before flagging the
        stopping condition. An instance of this class is `callable`, and is to be used as follows:
    :param lower_is_better: if True (default) the metric is to be minimized.
    :ivar best_score: keeps track of the best value seen so far
    :ivar best_epoch: keeps track of the epoch in which the best score was set
    :ivar STOP: flag (boolean) indicating the stopping condition
    :ivar IMPROVED: flag (boolean) indicating whether there was an improvement in the last call
    """

    def __init__(self, patience, lower_is_better=True):

        self.PATIENCE_LIMIT = patience
        self.better = lambda a,b: a<b if lower_is_better else a>b
        self.patience = patience
        self.best_score = None
        self.best_epoch = None
        self.STOP = False
        self.IMPROVED = False

    def __call__(self, watch_score, epoch):
        """
        Commits the new score found in epoch `epoch`. If the score improves over the best score found so far, then
        the patiente counter gets reset. If otherwise, the patience counter is decreased, and in case it reachs 0,
        the flag STOP becomes True.

        :param watch_score: the new score
        :param epoch: the current epoch
        """
        self.IMPROVED = (self.best_score is None or self.better(watch_score, self.best_score))
        #print(watch_score)
        #print(self.best_score)
        #print(self.IMPROVED)
        if self.IMPROVED:
            self.best_score = watch_score
            self.best_epoch = epoch
            self.patience = self.PATIENCE_LIMIT
        else:
            self.patience -= 1
            if self.patience <= 0:
                self.STOP = True

def set_memory_limit(limit_mb):
    resource.setrlimit(resource.RLIMIT_AS, (limit_mb * 1024 * 1024, resource.RLIM_INFINITY))

# Set the memory limit to 2 GB (2000 MB)
#set_memory_limit(5000)


def phonemize_sentence(sentence, language):
    return phonemize(sentence, language=language)

def process_and_phonemize_batch(dataset_filename, new_dataset, language, batch_size=1000):
    with open(dataset_filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    sentences = []
    labels = []
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            label, sentence = parts
            sentences.append(sentence)
            labels.append(label)

    total_sentences = len(sentences)
    batches = (total_sentences + batch_size - 1) // batch_size

    for batch_num in range(batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_sentences)
        batch_sentences = sentences[start_idx:end_idx]

        num_processes = cpu_count()
        pool = Pool(processes=num_processes)
        phonemization_function = partial(phonemize_sentence, language=language)

        phonemized_results = []
        with tqdm(total=len(batch_sentences), desc=f"Phonemizing Batch {batch_num+1}/{batches}", unit="sentence") as pbar:
            for result in pool.imap_unordered(phonemization_function, batch_sentences):
                phonemized_results.append(result)
                pbar.update(1)

        with open(new_dataset, 'a', encoding='utf-8') as file:
            for label, result in zip(labels[start_idx:end_idx], phonemized_results):
                file.write(f'{label}\t{result}\n')

        pool.close()
        pool.join()
    
def save_phonemized_dataset(dataset_filename, phonemized_sentences, phonemized_path):
    with open(phonemized_path, 'w', encoding='utf-8') as file:
        with open(dataset_filename, 'r', encoding='utf-8') as original_file:
            original_lines = original_file.readlines()

            for i, line in enumerate(original_lines):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    label = parts[0]
                    phonemized_sentence = phonemized_sentences[i]
                    new_line = f"{label}\t{phonemized_sentence}\n"
                    file.write(new_line)


if __name__ == '__main__':
    from data.data import TextDatasetHate
    import time

    #data = TextDatasetHate()
    #X = data.train_X

    #char_vocab = CharVocabulary(X)
    #word_vocab = WordVocabulary(X)
    #start_time = time.time()
    #phonetic_char_vocab = PhoneticCharVocabulary(X,
    #                                              '/home/sperduti/sound_embeddings/sound_embeddings_utils/phonetic_subwords_dataset/hate/phonetic_dict.csv',
    #                                              '/home/sperduti/sound_embeddings/Datasets/hate/phonetic_vocab',
    #                                              use_existing=False)
    #end_time = time.time()
    #print("Execution time:", end_time - start_time, "seconds")

    #print(phonetic_char_vocab.encode('Hello, how are you!?'))

    dataset_filename = 'sound_embeddings/Datasets/hate/misspelled_datasets/test.txt'  # Replace with your dataset filename
    language = 'en-us'  # Replace with the desired language code
    phonemized_path = 'sound_embeddings/Datasets/hate/phonemized_dataset/test_misspelled.txt'

    process_and_phonemize_batch(dataset_filename, phonemized_path, language, 100)


