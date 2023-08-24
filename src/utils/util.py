from abc import ABC, abstractmethod

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


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


class CharVocabulary(Vocabulary, Encoder):
    def __init__(self, elements):
        super().__init__(' '.join(elements))

    def encode(self, instance):
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


if __name__ == '__main__':
    from data import TextDatasetHate

    data = TextDatasetHate()
    X = data.train_X

    char_vocab = CharVocabulary(X)
    word_vocab = WordVocabulary(X)

    print(f'{len(word_vocab) = }')

    print(char_vocab.encode('Hello, how are you!?'))
    print(word_vocab.encode('Hello, how are you werwerwer!?'))

