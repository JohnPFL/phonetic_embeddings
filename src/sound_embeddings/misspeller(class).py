import random
import pandas as pd
from phonemizer import phonemize


class PhoneticMisspeller:
    def __init__(self, dataframe):
        self.dataframe = pd.read_csv(dataframe, delimiter=';')
        self.misspellings_dict = {}

    def exclude_and_get_random(self, L, x):
        if x in L:
            L.remove(x)
        if L:
            random_entry = random.choice(L)
            return random_entry
        else:
            return None

    def has_substring_in_list(self, x, L):
        list_of_subwords = []
        for word in L:
            if word in x:
                list_of_subwords.append(word)
        return list_of_subwords

    def morph_substitution(self, morhpeme, phoneme):
        subwords_selection = list(self.dataframe['subword'][self.dataframe['phonetic'] == phoneme])
        for subword in subwords_selection:
            if subword in morhpeme:
                rand_selection = self.exclude_and_get_random(subwords_selection, subword)
                if rand_selection is None:
                    return morhpeme
                new_morpheme = morhpeme.replace(subword, rand_selection)
                return new_morpheme
            else:
                continue
        return morhpeme
    
    def create_mapping_phoneme_to_word(self, phonetic_transcriptions, word_list):
        return {ph: wd for ph, wd in zip(phonetic_transcriptions, word_list)}
    
    def process_phonetic_mapping(self, phonetic, word):
        phonetic_mappings = self.dataframe.loc[self.dataframe['phonetic'] == phonetic, 'subword']
        if len(phonetic_mappings) > 0: # type: ignore
            if len(phonetic_mappings) == 1 and phonetic_mappings.iloc[0] != word: # type: ignore
                self.misspellings_dict[phonetic_mappings.iloc[0]] = word # type: ignore
            else:
                new_morpheme = self.exclude_and_get_random(phonetic_mappings.tolist(), word) # type: ignore
                self.misspellings_dict[word] = new_morpheme if new_morpheme else word
        else:
            self.process_no_direct_mapping(phonetic, word)

    def process_no_direct_mapping(self, phonetic, word):
        all_phonemes = self.dataframe['phonetic'].tolist()
        selected_phonemes = self.has_substring_in_list(phonetic, all_phonemes)
        sorted_phonemes = sorted(selected_phonemes, key=len, reverse=True)
        new_morph = None

        for phoneme in sorted_phonemes:
            if new_morph is None or new_morph == word:
                new_morph = self.morph_substitution(word, phoneme)
            else:
                self.misspellings_dict[word] = new_morph
                break

    def misspell_substitutions(self, sentence):
        words = sentence.split()
        misspelled_words = [self.misspellings_dict.get(word, word) for word in words]
        misspelled_sentence = ' '.join(misspelled_words)
        return misspelled_sentence

    def process_sentence(self, sentence):
        phonetic_transcriptions = self.transcribe_sentence(sentence).replace('ː', '') # type: ignore
        word_list = sentence.split()
        phonetic_transcriptions = phonetic_transcriptions.split()

        mapping_phoneme_to_word = self.create_mapping_phoneme_to_word(phonetic_transcriptions, word_list)

        for phonetic, word in mapping_phoneme_to_word.items():
            self.process_phonetic_mapping(phonetic, word)

        return self.misspellings_dict
    
    def misspell_sentence(self, sentence):
        self.misspellings_dict = self.process_sentence(sentence)
        misspelled_sentence = self.misspell_substitutions(sentence)
        return misspelled_sentence
    
    def misspell_dataset(self, dataset):
        return [self.misspell_sentence(sentence) for sentence in dataset]

    def transcribe_sentence(self, sentence):
        # Replace this with your phonemize logic
        phonetic_transcriptions = phonemize(sentence)
        # For now, I'll just return a dummy value
        return phonetic_transcriptions


# Example usage:

#csv_path = '/home/sperduti/sound_embeddings/sound_embeddings_utils/phonetic_subwords_dataset/hate/phonetic_dict.csv'
# try with hate speech and see if this is working
#sentence = ["my name is gianluca", "you are beautiful", "these niggers should die"]
#phoneme_substitution = PhoneticMisspeller(csv_path)
#misspellings_dict = phoneme_substitution.misspell_dataset(sentence)
#print(misspellings_dict)