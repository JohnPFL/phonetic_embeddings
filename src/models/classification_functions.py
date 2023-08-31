
import os
from os.path import join
import sys
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

from utils.util import EarlyStop
from data.data import  NeuralDataset
from data import data



def evaluate_metrics(predictions, true_labels, dataset=None):
    """
    Calculate various metrics based on predictions and true labels.

    Args:
        predictions (array-like): Predicted labels.
        true_labels (array-like): True labels.
        dataset: Test dataset containing true labels (optional).

    Returns:
        dict: Dictionary of calculated metrics.
    """
    if dataset is not None:
        test_y = dataset.test_y
    else:
        test_y = true_labels
    
    precision = round(precision_score(test_y, predictions), 3) # type: ignore
    recall = round(recall_score(test_y, predictions), 3) # type: ignore
    macrof1 = round(f1_score(test_y, predictions, average='macro'), 3) # type: ignore
    f1 = round(f1_score(test_y, predictions), 3) # type: ignore
    f1neg = round(f1_score(test_y, predictions, pos_label=0), 3) # type: ignore
    acc = round((test_y == predictions).mean(), 3)
    
    results = {
        'precision': precision,
        'recall': recall,
        'macrof1': macrof1,
        'f1': f1,
        'f1neg': f1neg,
        'accuracy': acc
    }
    return results

def validate_batch(model, criterion, X_batch, y_batch, device='cuda'):
    """
    Perform validation on a batch of data using the model.

    Args:
        model (torch.nn.Module): The model to be validated.
        criterion: The loss criterion.
        X_batch: Input data.
        y_batch: True labels.
        device (str): Device to perform calculations on.

    Returns:
        tuple: Loss, predictions.
    """
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    output = model(X_batch)
    loss = criterion(output, y_batch)
    predictions = (torch.sigmoid(output) > 0.5)
    return loss, predictions

def train_batch(model, optimizer, criterion, X_batch, y_batch, device='cuda'):
    """
    Perform a training step on a batch of data.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer: The optimizer to update model parameters.
        criterion: The loss criterion.
        X_batch: Input data.
        y_batch: True labels.
        device (str): Device to perform calculations on.

    Returns:
        torch.Tensor: Loss for the batch.
    """
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    output = model(X_batch)
    loss = criterion(output, y_batch)
    loss.backward()
    optimizer.step()
    return loss

def train_epoch(model, train_loader, optimizer, criterion, epoch, device='cuda', batch_size=4):
    """
    Perform training for an epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader: Data loader for training data.
        optimizer: The optimizer to update model parameters.
        criterion: The loss criterion.
        epoch (int): Current epoch.
        device (str): Device to perform calculations on.
        batch_size (int): Batch size.

    Returns:
        float: Mean training loss for the epoch.
    """
    model.train()
    loss_history = []
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (X_batch, y_batch) in pbar:
        if X_batch.size(0) >= batch_size:
            loss = train_batch(model, optimizer, criterion, X_batch, y_batch, device)
            loss_history.append(loss.item())
            pbar.set_description(
                f'Train {epoch = } [{batch_idx * len(X_batch)}/{len(train_loader.dataset)} '
                f'({100. * batch_idx / len(train_loader):.1f}%) Loss {np.mean(loss_history):.6f}]')
    return np.mean(loss_history)

def val_epoch(model, valid_loader, criterion, valid_y, device='cuda', batch_size=4):
    """
    Validate the model on the validation dataset.

    Args:
        model (torch.nn.Module): The model to be validated.
        valid_loader: Data loader for validation data.
        criterion: The loss criterion.
        valid_y: True labels for validation data.
        device (str): Device to perform calculations on.
        batch_size (int): Batch size.

    Returns:
        float: Mean validation loss.
    """
    with torch.no_grad():
        loss_history = []
        all_predictions = []
        for batch_idx, (X_batch, y_batch) in enumerate(valid_loader):
            if X_batch.size(0) < batch_size:
                continue
            loss, predictions = validate_batch(model, criterion, X_batch, y_batch, device)
            all_predictions.append(predictions.to('cpu').numpy())
            loss_history.append(loss.item())
        all_predictions = np.concatenate(all_predictions)
        valid_y = valid_y[:len(all_predictions)]
        metrics = evaluate_metrics(all_predictions, valid_y)
        macrof1, f1, f1neg = metrics['macrof1'], metrics['f1'], metrics['f1neg']
        print(f'\nValidation Loss {np.mean(loss_history):.6f}  macro-f1={macrof1:.4f} f1={f1:.4f}  f1neg={f1neg:.4f}')
        return np.mean(loss_history)

def predict_samples(model, samples, char_encoder, batch_size, lazy_loader=False, device='cuda'):
    """
    Predict labels for a set of samples using the provided model.

    Args:
        model: The trained model for predictions.
        samples: Input data samples.
        char_encoder: Character encoder.
        batch_size: Batch size for predictions.
        lazy_loader: Lazy encoding flag.
        device: Device to perform calculations on.

    Returns:
        numpy.ndarray: Predicted labels for the samples.
    """
    data_loader = NeuralDataset(samples, None, char_encoder, lazy_encoding=lazy_loader).asDataLoader(batch_size)
    all_predictions = []
    with torch.no_grad():
        for batch_idx, X_batch in enumerate(data_loader):
            X_batch = X_batch.to(device)
            output = model(X_batch)
            predictions = (torch.sigmoid(output) > 0.5)
            all_predictions.append(torch.asarray(predictions.to('cpu')))
    return np.concatenate(all_predictions)

def predict_datasets(model, test_dataset_adv, test_dataset_clean):
    """
    Predict labels for adversarial and clean test datasets using the provided model.

    Args:
        model: The trained model for predictions.
        test_dataset_adv: Adversarial test dataset.
        test_dataset_clean: Clean test dataset.

    Returns:
        tuple: Predicted labels for adversarial and clean datasets.
    """
    predictions_adv = model.predict(test_dataset_adv.test_X)
    predictions_clean = model.predict(test_dataset_clean.test_X)
    return predictions_adv, predictions_clean

def weights_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)

def initialize_early_stop(patience):
    """
    Initialize EarlyStop object if patience is provided.

    Args:
        patience (int): The number of epochs with no improvement after which training will be stopped.

    Returns:
        EarlyStop or None: EarlyStop object if patience is non-zero, else None.
    """
    if patience:
        return EarlyStop(patience)
    return None

def definig_latest_model(dataset_name, model_name, epoch):
    timestamp = time.time()
    formatted_time = time.strftime('%Y-%m-%d%H:%M:%S', time.localtime(timestamp))
    latest_model = f"{dataset_name}{model_name.split('/')[-1]}_epoch_{epoch}_time_{formatted_time}.pt"
    return latest_model

def training_loop(model, train_X, train_y, valid_y, 
                  valid_loader, optimizer, 
                  criterion, early_stop, 
                  nepochs, max_instances,
                  tmp_path, def_path, dataset_name, 
                  model_name, device, batch_size, char_encoder,
                  max_length, lazy_loader):
    """
    Main training loop.

    Args:
        model: The neural network model to train.
        train_X (array-like): Training data features.
        train_y (array-like): Training data labels.
        valid_y (array-like): Validation data labels.
        valid_loader: DataLoader for validation data.
        optimizer: The optimizer for updating model parameters.
        criterion: Loss criterion.
        early_stop: EarlyStop object for early stopping.
        nepochs (int): Number of epochs to train.
        max_instances (int): Maximum instances per training subset.
        tmp_path (str): Path to temporary storage.
        def_path (str): Path to store the final model.
        dataset_name (str): Name of the dataset.
        model_name (str): Name of the model.
        device (str): Device to use for training.
        batch_size (int): Batch size for training.
        char_encoder: Character encoder.
        max_length (int): Maximum sequence length.
        lazy_loader (bool): Lazy encoding flag.
    """
    latest_model = None
    for epoch in range(nepochs):
        start_index = (epoch % (len(train_X) // max_instances)) * max_instances
        end_index = start_index + max_instances
        subset_X, subset_y = train_X[start_index:end_index], train_y[start_index:end_index]

        train_loader = NeuralDataset(subset_X, subset_y, char_encoder, 
                                     MAX_LENGTH=max_length, 
                                     lazy_encoding=lazy_loader).asDataLoader(batch_size, shuffle=True)

        train_loss = train_epoch(model, train_loader, optimizer, 
                                 criterion, epoch, device=device, batch_size=batch_size)
        val_loss = val_epoch(model, valid_loader, criterion, 
                             valid_y, device=device, batch_size=batch_size)

        if early_stop:
            if latest_model is None:
                latest_model = definig_latest_model(dataset_name, 
                                                model_name,
                                                epoch)
            latest_model = process_early_stop(model, val_loss, 
                                                    epoch, early_stop, 
                                                    tmp_path, def_path,   
                                                    dataset_name, 
                                                    model_name,
                                                    latest_model, valid_loader, 
                                                    optimizer, criterion, device)
        else:
            break
        
        if early_stop.STOP is True:
            return def_path, latest_model

    return def_path, latest_model



def process_early_stop(model, val_loss, epoch, early_stop, tmp_path, 
                       def_path, 
                        dataset_name, 
                                                model_name, latest_model, 
                       valid_loader, optimizer, criterion, device):
    """
    Process early stopping and save the best model.

    Args:
        model: The neural network model.
        val_loss (float): Validation loss.
        epoch (int): Current epoch.
        early_stop: EarlyStop object for early stopping.
        tmp_path (str): Path to temporary storage.
        def_path (str): Path to store the final model.
        dataset_name (str): Name of the dataset.
        model_name (str): Name of the model.
        valid_loader: DataLoader for validation data.
        optimizer: The optimizer for updating model parameters.
        criterion: Loss criterion.
        device (str): Device to use for training.

    Returns:
        str: Path to the latest saved model.
    """
    early_stop(val_loss, epoch)
    print(f'improved: {early_stop.IMPROVED}')
    print(tmp_path)
    if early_stop.IMPROVED:
        latest_model = definig_latest_model(dataset_name, model_name, epoch)
        torch.save(model, join(tmp_path, latest_model))
    elif early_stop.STOP:
        shutil.move(join(tmp_path, latest_model), join(def_path, latest_model))
        for filename in os.listdir(tmp_path):
            file_path = join(tmp_path, filename)
            if filename != latest_model and os.path.isfile(file_path):
                os.remove(file_path)
        model = torch.load(join(def_path, latest_model))
        train_epoch(model, valid_loader, optimizer, criterion, epoch + 1, device=device)
        torch.save(model, join(def_path, latest_model))
        latest_model = join(def_path, latest_model)
        return latest_model

    return latest_model

def train_model(model, X, y, char_encoder, dataset_name, batch_size, seed,
                max_length=100, tmp_path='',
                def_path='', nepochs=500,
                lazy_loader=False, device='cuda', model_name='model',
                patience=10, lr=0.0001):
    """
    Train the model using the provided configuration.

    Args:
        model: The neural network model to train.
        X: Training data features.
        y: Training data labels.
        char_encoder: Character encoder.
        dataset_name (str): Name of the dataset.
        batch_size (int): Batch size for training.
        seed (int): Random seed for reproducibility.
        max_length (int): Maximum sequence length.
        tmp_path (str): Path to temporary storage.
        def_path (str): Path to store the final model.
        nepochs (int): Number of epochs to train.
        lazy_loader (bool): Lazy encoding flag.
        device (str): Device to use for training.
        model_name (str): Name of the model.
        patience (int): The number of epochs with no improvement after which training will be stopped.
        lr (float): Learning rate for the optimizer.

    Returns:
        str: Path to the final saved model.
    """

    max_instances = 1819

    val_size = int(len(X) * 0.25)
    if val_size > 2000:
        val_size = 2000

    train_X, valid_X, train_y, valid_y = train_test_split(X, y, stratify=y, 
                                                          test_size=val_size, random_state=seed)
    
    valid_loader = NeuralDataset(valid_X, valid_y, char_encoder, 
                                 MAX_LENGTH=max_length, lazy_encoding=lazy_loader).asDataLoader(batch_size)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    early_stop = initialize_early_stop(patience)


    def_path, latest_model = training_loop(model, train_X, train_y, valid_y, 
                                           valid_loader, optimizer, 
                                            criterion, early_stop, 
                                            nepochs, max_instances,
                                            tmp_path, def_path, 
                                            dataset_name, model_name, 
                                            device, batch_size, 
                                            char_encoder, max_length, lazy_loader)

    if not latest_model:
        latest_model = 'BUG'

    return join(def_path, latest_model)

def prepare_dataframe(dataframe_path, dataset_name, 
                      condition, implied_models, 
                      implied_metrics, rerun_check):
    """
    Prepare an empty or loaded DataFrame for storing experiment results.

    Args:
        dataframe_path (str): Path to the DataFrame pickle file.
        dataset_name (str): Name of the dataset.
        hardness (str or int): Hardness level of the dataset.
        condition (str): Condition of the experiment (e.g., 'adversarial', 'clean').
        implied_models (list or Model): List of implied model names or a single implied model.
        implied_metrics (list): List of implied metrics.
        rerun_check (bool): Flag to check if experiments should be rerun.

    Returns:
        pd.DataFrame: Prepared DataFrame for storing experiment results.
    """
    # If the DataFrame file doesn't exist, create a new DataFrame structure
    if not os.path.exists(dataframe_path):
        # Create column index for MultiIndex DataFrame
        if isinstance(implied_models, list):
            a_columns = pd.MultiIndex.from_product([[f"{dataset_name}"], [condition], implied_models])
        else: 
            a_columns = pd.MultiIndex.from_product([[f"{dataset_name}"], [condition], [implied_models.name]])
        df = pd.DataFrame(index=implied_metrics, columns=a_columns)
    else:
        # Load the existing DataFrame
        df = pd.read_pickle(dataframe_path)
        
        # Check if rerun_check is True and if any NaN values exist in the DataFrame
        if rerun_check and df.isnull().any().any():
            print(f'Experiments already ran. Check the results at {dataframe_path}')
            sys.exit()
    
    return df


def add_results_to_dataframe(df, dataset_name, condition, model_name, implied_metrics, results):
    """
    Add experiment results to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to which results will be added.
        dataset_name (str): Name of the dataset.
        hardness (str or int): Hardness level of the dataset.
        condition (str): Condition of the experiment (e.g., 'adversarial', 'clean').
        model_name (str): Name of the model.
        implied_metrics (list): List of implied metrics.
        results (dict): Dictionary containing experiment results.

    Returns:
        pd.DataFrame: DataFrame with added experiment results.
    """
    for metric in implied_metrics:
        try:
            # Try to add the metric result to the DataFrame cell
            df.loc[metric, (f"{dataset_name}", condition, model_name)] = results[metric]
        except KeyError:
            # If the metric is missing in the results, fill the cell with 0
            df.loc[metric, (f"{dataset_name}", condition, model_name)] = 0
    
    return df


def prepare_and_update_dataframe(dataframe_path_a, dataframe_path_c, type_of_adv, dataset_name,
                                 implied_models, implied_metrics, rerun_check, model_name, 
                                 results_clean, results_adversarial):
    # Prepare or load DataFrames for adversarial and clean conditions
    df_adv = prepare_dataframe(dataframe_path_a, dataset_name, type_of_adv, implied_models, implied_metrics, rerun_check)
    df_c = prepare_dataframe(dataframe_path_c, dataset_name, 'clean', implied_models, implied_metrics, rerun_check)
    
    # Add results to the clean DataFrame
    df_c = add_results_to_dataframe(df_c, dataset_name,'clean', model_name, implied_metrics, results_clean)
    
    # Add results to the adversarial DataFrame
    df_adv = add_results_to_dataframe(df_adv, dataset_name,  type_of_adv, model_name, implied_metrics, results_adversarial)
    
    return df_c, df_adv

def pipeline_adversarial(cln_path, adv_path, 
                         model, dataframe_path, 
                         verbose, rerun_check, implied_models, pht_path = 'False'):
    """
    Execute the pipeline for adversarial experiments.

    Args:
        cln_path (str): Path to clean data.
        adv_path (str): Path to adversarial data.
        model: The machine learning model to use.
        dataframe_path (str): Path for storing DataFrame pickle files.
        hardness: Hardness level of the dataset.
        kernel_heights: Kernel heights for the model.
        visual_kernel_heights: Visual kernel heights for the model.
        seed (int): Random seed.
        verbose (bool): Flag to enable verbose printing.
        rerun_check (bool): Flag to check if experiments should be rerun.
        implied_models: List of implied model names.

    Returns:
        None
    """
    
    if pht_path == 'False':
        pht_path = False

    dataset_clean = data.MultipathDataset(cln_path, cln_path, phonetic = True)
    dataset_adv = data.MultipathDataset(cln_path, adv_path, phonetic = True)

    if pht_path:
        dataset_pht = data.MultipathDataset(cln_path, pht_path)

    # Datasetname is extrapolated from the given clean path
    dataset_name = cln_path.split('/')[-1]
    dataframe_path_c = dataframe_path + '_cl.pkl'
    dataframe_path_a = dataframe_path + '_adv.pkl'

    if pht_path:
        dataframe_path_pht = dataframe_path + 'pht.pkl'

    if model.name != 'svm':
        model.fit(dataset_clean.train_X, dataset_clean.train_y, dataset_name)
    else:
        model.fit(dataset_clean.train_X, dataset_clean.train_y)

    predictions_adv, predictions_clean = predict_datasets(model, dataset_adv, dataset_clean)

    if pht_path:
        predictions_pht = model.predict(dataset_pht.test_X) # type: ignore

    # _c == metrics for the clean test dataset
    results_c = evaluate_metrics(dataset_clean.test_y, predictions_clean)
    # _a == metrics for the adversarial test dataset
    results_a = evaluate_metrics(dataset_adv.test_y, predictions_adv)

    if pht_path:
        results_pht = evaluate_metrics(predictions_pht, dataset_pht.test_y) # type: ignore
        print('these are the results with phonemic dataset:', results_pht)
    
    print('these are the results for clean:', results_c)
    print('these are the results for adversarial:', results_a)
    

    implied_metrics = ['precision',  'macrof1',  'f1', 'f1neg', 'recall',  'accuracy']

    type_of_adv = 'adversarial'

    df_c, df_adv = prepare_and_update_dataframe(dataframe_path_a, dataframe_path_c, type_of_adv,
                                                dataset_name,  
                                                implied_models, implied_metrics, 
                                                rerun_check, model.name, results_c, results_a)
    if pht_path: 
        type_of_adv = 'phonetized_adversarial'
        df_c, df_pht= prepare_and_update_dataframe(dataframe_path_a, dataframe_path_c, type_of_adv,
                                                dataset_name,  
                                                implied_models, implied_metrics, 
                                                rerun_check, model.name, results_c, results_a)

    df_c.to_pickle(dataframe_path_c)
    df_adv.to_pickle(dataframe_path_a)
    if pht_path: #type: ignore
        df_pht.to_pickle(dataframe_path_pht) #type: ignore
    
    # Print dataframes if verbose
    if verbose:
        print("Clean DataFrame:")
        print(df_c)
        print("Adversarial DataFrame:")
        print(df_adv)
        if df_pht is not None: #type: ignore
            print("Phonetized adversarial")
            print(df_pht) #type: ignore