import random

from phonemizer import phonemize
import pandas as pd

csv = '/home/sperduti/sound_embeddings/sound_embeddings_utils/phonetic_subwords_dataset/hate/phonetic_dict.csv'

def transcribe_sentence(sentence):
    phonetic_transcriptions = phonemize(sentence)
    return phonetic_transcriptions

def exclude_and_get_random(L, x):
    if x in L:
        L.remove(x)
    if L:
        random_entry = random.choice(L)
        return random_entry
    else:
        return None  # Return None if the list is empty

def morph_substitution(morhpeme, phoneme, dataset):
    subwords_selection = list(dataset['subword'][dataset['phonetic']==phoneme])
    for subword in subwords_selection:
        if subword in morhpeme:
            rand_selection = exclude_and_get_random(subwords_selection, subword)
            if rand_selection is None:
                return morhpeme
            new_morpheme = morhpeme.replace(subword, rand_selection)
            return new_morpheme
        else:
            continue
    return morhpeme


# QUI VA COSTRUITA UNA LISTA
def has_substring_in_list(x, L):
    list_of_subwords = []
    for word in L:
        if word in x:
            list_of_subwords.append(word)
    return list_of_subwords

#def phoneme_check(phon_to_morph_dict, phon_subwords_df):
    #mapping_phoneme_to_word

#def morphology_check(morph_to_phon_dict, phon_subwords_df):

# Example usage:
sentence = "my ass stinks ahah"
phonetic_transcriptions = transcribe_sentence(sentence)
phonetic_transcriptions = phonetic_transcriptions.replace('ː', '') # type: ignore

word_list = sentence.split()
phonetic_transcriptions = phonetic_transcriptions.split()

mapping_phoneme_to_word = {}
word_to_phoneme = {}
for word in range(len(phonetic_transcriptions)):
    mapping_phoneme_to_word[phonetic_transcriptions[word]] = word_list[word] 
    word_to_phoneme[word_list[word]] = phonetic_transcriptions[word]


csv = pd.read_csv(csv, delimiter=';')



misspellings_dict = {}
for wp in mapping_phoneme_to_word.items():
    print(wp)
    # corrispondenze morfologiche che suonano ugualmente tra loro (di solito)
    phonetic_mappings = csv['subword'][csv['phonetic']==wp[0]]
    # se non ci sono corrispondenze dirette (ex. "i" -> "ai"), si prova un altro metodo (vedasi punto (2)).
    # Altrimenti (1), segue descrizione...
    if len(list(phonetic_mappings)) > 0:
        if len(list(phonetic_mappings)) == 1:
            # se c'è solo una diretta corrispondenza fonetica, si controlla se la parte morfologica è identica. In caso lo sia, si va avanti
            if list(phonetic_mappings)[0] == wp[1]:
                pass
            else:
                # altrimenti, si inserisce una prima "entry" nel dizionario delle sostituzioni, perché 
                misspellings_dict[list(phonetic_mappings)[0]] = wp[1]
        else: 
            # se la corrispondenza morfologica non è univoca, avremo qui una lista di morfemi corrispondenti. Dobbiamo escludere il nostro morfema 
            # originale e scegliere uno randomico tra gli altri
            new_morpheme = exclude_and_get_random(list(phonetic_mappings), wp[1])
            misspellings_dict[wp[1]] = new_morpheme
            print('here we should exclude the morphology part which is already assigned to the phoneme and take another random one')
    else: 
        # (2) si controlla se una delle parti della parola è all'interno del nostro dizionario delle sostituzioni
        # quasi sempre questo avverrà: seastar non ha corrispondenza diretta, ma magari c'è la parola sea che ha delle corrispondenze.
        # In questo caso, si sfrutta questa informazione. Si parte dalle corrispondenze più grandi e man mano si arriva anche alle singole lettere.
        all_phonemes = list(csv['phonetic'])
        # Qui si selezionano i subphonemes che fanno parte del nostro gruppo fonetico
        selected_phonemes = has_substring_in_list(wp[0], all_phonemes)
        # Si mettono in ordine di grandezza
        sorted_phonemes = sorted(selected_phonemes, key=lambda x: len(x), reverse=True)
        # Questa è una variabile di appoggio per checking della morph_substitution
        new_morph = None
        for phoneme in sorted_phonemes:
            if new_morph == wp[1] or new_morph is None:
                # si controlla la corrispondenza tra parti morfologiche e fonemi. Ad esempio, somebody. "body" è presente nella lista 
                # dei morfemi? Sì. Nella parola? Sì. Si procede quindi con una sostituzione con una delle alternative fonetiche (bodie, body)
                new_morph = morph_substitution(wp[1], phoneme, csv)
            else:
                misspellings_dict[wp[1]] = new_morph
                break



words = sentence.split()

corrected_words = []
for word in words:
    corrected_word = misspellings_dict.get(word, word)  # Use the correction if available, otherwise use the original word
    corrected_words.append(corrected_word)

corrected_sentence = ' '.join(corrected_words)

print("Original sentence:", sentence)
print("Phonetic transcriptions:", corrected_sentence)