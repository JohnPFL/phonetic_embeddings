import csv
import os

from phonetic_dict_builder import PhoneticDictionaryBuilder

class PhoneticLabeling:
    def __init__(self, csv_file_path, phonetic_data):
        self.csv_file_path = csv_file_path
        self.fieldnames = ['subword', 'phonetic']
        self.phonetic_data = phonetic_data

    @staticmethod
    def exclude_words_from_csv(csv_file_path):
        excluded_words = set()

        if not csv_file_path:
            return excluded_words

        try:
            with open(csv_file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=';')
                for row in reader:
                    excluded_words.add(row['subword'])
        except FileNotFoundError:
            pass  # File not found, no words to exclude

        return excluded_words

    def label_data(self):

        # Write the header if the CSV file doesn't exist
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames, delimiter=';')
                writer.writeheader()

        print("Type 'end()' to finish or follow the prompts.")
        
        for word, phonetic in self.phonetic_data.items():
            prompt = f"{word} -> {phonetic}\nEnter the phonetic breakdown (end() to finish): "
            breakdown = '0'
            for rep in range(10):
                if breakdown[0] != 'end()':
                    breakdown = input(prompt)
                    if breakdown == 'end()':
                        break

                    breakdown = breakdown.split(';')
                    if len(breakdown) == 2:
                        with open(self.csv_file_path, 'a', newline='') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames, delimiter=';')
                            writer.writerow({'subword': breakdown[0], 'phonetic': breakdown[1]})
                    else:
                        print("Invalid input format. Please provide subword and phonetic separated by ';'")

        print("Data has been written to", self.csv_file_path)

def main():
    dataset_path = '/home/sperduti/sound_embeddings/Datasets/hate/train.txt'
    phonetic_dict_file_path = '/home/sperduti/sound_embeddings/sound_embeddings_utils/phonetic_dicts/hate/hate_phonetic_embs.pickle'
    label_csv_file_path = '/home/sperduti/sound_embeddings/sound_embeddings_utils/phonetic_subwords_dataset/hate/phonetic_dict.csv'
    max_words = 2000

    phonetic_dict_builder = PhoneticDictionaryBuilder(dataset_path, max_words)
    # The phonemization can be done from zero, or it can be loaded. In this case we want to load it.
    phonetic_dict = phonetic_dict_builder.load_dictionary_from_pickle(phonetic_dict_file_path)
    pdb = PhoneticLabeling(label_csv_file_path, phonetic_dict)

    pdb.label_data()
                        

if __name__ == '__main__':
    main()