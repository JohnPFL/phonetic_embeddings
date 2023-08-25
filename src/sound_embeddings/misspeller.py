import random
import pandas as pd
from phonemizer import phonemize
import multiprocessing
from functools import partial
from phonemizer import phonemize
from tqdm import tqdm  # Import the tqdm library
import string
import os

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
    
    def process_batch(self, batch):
        return [self.misspell_sentence(sentence) for sentence in batch]
    
    def _load_existing_dataset(self, save_folder):
        existing_dataset_path = os.path.join(save_folder, "test.txt")
        if os.path.exists(existing_dataset_path):
            with open(existing_dataset_path, "r") as f:
                batched_results = f.read().splitlines()
            return batched_results
        return None

    def _save_dataset(self, save_folder, batched_results):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        dataset_path = os.path.join(save_folder, "test.txt")
        with open(dataset_path, "w") as f:
            for result in batched_results:
                f.write(result + "\n")

    
    def batch_misspell_dataset_parallel(self, X, y, save_folder, batch_size=10):
        num_processes = multiprocessing.cpu_count()  # Get the number of available CPU cores
        total_batches = len(X) // batch_size + (1 if len(X) % batch_size != 0 else 0)
        
        batched_results = self._load_existing_dataset(save_folder)
        if batched_results is not None:
            return batched_results
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            batched_results = []
            for batch_num, i in enumerate(range(0, len(X), batch_size), start=1):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                # Apply misspelling only to the sentences with positive labels (1)
                batch_results = []
                for sentence, label in zip(batch_X, batch_y):
                    if label == '1':
                        batch_results.append(self.misspell_sentence(sentence))
                    else:
                        batch_results.append(sentence)
                batched_results.extend(batch_results)

                print(f"Processed batch {batch_num}/{total_batches}")
        
        if save_folder:
            self._save_dataset(save_folder, batched_results)
        
        return batched_results

    def misspell_entire_dataset(self, X, y, batch_size=10):
        batched_dataset = self.batch_misspell_dataset_parallel(X, y, batch_size)
        return batched_dataset

    def transcribe_sentence(self, sentence):
        # Replace this with your phonemize logic
        phonetic_transcriptions = phonemize(sentence)
        # For now, I'll just return a dummy value
        return phonetic_transcriptions
    
    ##########TEST
    

    def parallel_phonemize(self, sentence, language):
        return phonemize(sentence, language=language)
    
    def remove_non_alphanumeric(self, text):
        translator = str.maketrans('', '', string.punctuation.replace(' ', ''))
        return text.translate(translator)

    def batch_phonemize_parallel(self, sentences, language, verbose=True):
        num_processes = multiprocessing.cpu_count()  # Get the number of available CPU cores
        pool = multiprocessing.Pool(processes=num_processes)
        phonemization_function = partial(self.parallel_phonemize, language=language)
        # we only need transcription for unique words
        sentences = list(set(self.remove_non_alphanumeric(sentences).split()))
        
        if verbose:
            with tqdm(total=len(sentences), desc="Phonemizing sentences", unit="sentence") as pbar:
                phonemized_results = []
                for result in pool.imap_unordered(phonemization_function, sentences):
                    phonemized_results.append(result)
                    pbar.update(1)
            pool.close()
            pool.join()

        else:
            phonemized_results = pool.map(phonemization_function, sentences)
            pool.close()
            pool.join()
            
        return phonemized_results


    ################


# Example usage:

#csv_path = '/home/sperduti/sound_embeddings/sound_embeddings_utils/phonetic_subwords_dataset/hate/phonetic_dict.csv'
# try with hate speech and see if this is working
#sentence = ["my name is gianluca", "you are beautiful", "these niggers should die"]
#phoneme_substitution = PhoneticMisspeller(csv_path)
#misspellings_dict = phoneme_substitution.misspell_dataset(sentence)
#print(misspellings_dict)