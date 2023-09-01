import sys
sys.path.append('/home/sperduti/sound_embeddings/src')
sys.path.append('/home/sperduti/sound_embeddings/src/models')
sys.path.append('/home/sperduti/sound_embeddings/src/module')
sys.path.append('/home/sperduti/sound_embeddings/src/utils')
sys.path.append('/home/sperduti/sound_embeddings/src/sound_embeddings')
sys.path.append('/home/sperduti/sound_embeddings/Datasets')

import argparse
from models.models_ import CNNbaseClassifier, SVMclassifier
from models.classification_functions import pipeline_adversarial

def main():
    parser = argparse.ArgumentParser(description="Run the pipeline.")
    parser.add_argument("--misspelled_dataset_path", required=True, help="Path to the misspelled dataset.")
    parser.add_argument("--results_path", required=True, help="Path to the results directory.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--embedding_dim", type=int, default=250, help="Embedding dimension.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--patience", type=int, default=40, help="Patience for early stopping.")
    parser.add_argument("--dataset_clean_path", required=True, help="Path to the clean dataset.")
    parser.add_argument("--tmp_model_path", required=True, help="Path to temporary model directory.")
    parser.add_argument("--def_model_path", required=True, help="Path to default model directory.")
    parser.add_argument("--phonemized_train", required=True, help="Path to phonemized training dataset.")
    parser.add_argument("--phonemized_misspelled_test", required=True, help="Path to phonemized misspelled test dataset.")
    args = parser.parse_args()
    
    path = {
        'tmp_path': args.tmp_model_path,
        'def_path': args.def_model_path
    }

    models = {
        'svm': SVMclassifier(),
        'cnnBase': CNNbaseClassifier(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            embedding_dim=args.embedding_dim,
            seed=args.seed,
            model_name='cnn_base',
            patience=args.patience,
            paths=path
        )
    }

    implied_models = ['cnn_phonetic']

    misspelled_dataset_path = args.misspelled_dataset_path
    phonemized_datasets = args.phonemized_train
    phonemized_misspelled_test = args.phonemized_misspelled_test

    pipeline_adversarial(
        phonemized_datasets,
        phonemized_datasets,
        models['cnnBase'], args.results_path,
        False, False, implied_models, phonemized_misspelled_test,
        phonetic = True, adversarial = True
    )

    pipeline_adversarial(
        args.dataset_clean_path,
        misspelled_dataset_path,
        models['cnnBase'], args.results_path,
        False, False, implied_models
    )

if __name__ == "__main__":
    main()
