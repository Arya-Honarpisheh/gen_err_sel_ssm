import numpy as np
import random

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import re


def create_majority_dataset(T, num_samples, batch_size=32, seed=0):
    ''' 
    Majority Dataset
    
    Args:
        T (int): Length of the input sequence.
        num_samples (int): Number of samples in the dataset.
        batch_size (int): Batch size for the DataLoader.
        seed (int): Seed for reproducibility.
    Returns:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
    '''

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    num_ones_train = np.random.randint(0, T+1, size=(num_samples))
    inputs_train = np.zeros((num_samples, T))
    for i in range(num_samples):
        inputs_train[i, 0:num_ones_train[i]] = 1
        np.random.shuffle(inputs_train[i])

    num_ones_test = np.random.randint(0, T+1, size=(num_samples))
    inputs_test = np.zeros((num_samples, T))
    for i in range(num_samples):
        inputs_test[i, 0:num_ones_test[i]] = 1
        np.random.shuffle(inputs_test[i])

    # Calculate outputs based on the majority of 1s
    outputs_train = (num_ones_train > (T / 2)).astype(int)
    outputs_test = (num_ones_test > (T / 2)).astype(int)

    # Add noise to the inputs
    noise_mask = (np.random.rand(num_samples, T) < 0.25) & (inputs_train == 1)
    inputs_train[noise_mask] = 0

    # noise_mask = np.random.rand(num_samples) < 0.1
    # outputs_train[noise_mask] = 1 - outputs_train[noise_mask]  # Flip the output with 10% probability

    # Convert inputs and outputs to torch tensors
    inputs_train = torch.tensor(inputs_train, dtype=torch.int32)
    outputs_train = torch.tensor(outputs_train, dtype=torch.int32)
    inputs_test = torch.tensor(inputs_test, dtype=torch.int32)
    outputs_test = torch.tensor(outputs_test, dtype=torch.int32)

    train_dataset = TensorDataset(inputs_train, outputs_train)
    test_dataset = TensorDataset(inputs_test, outputs_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_imdb(T, num_samples=1000, batch_size=16, seed=0):
    '''
    IMDB Large Movie Review Dataset from HuggingFace
    Args:
        T (int): Length of the input sequence.
        num_samples (int): Number of samples in the dataset.
        batch_size (int): Batch size for the DataLoader.
    Returns:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        tokenizer (transformers.AutoTokenizer): Tokenizer for the dataset.
    '''

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Load the entire dataset first (without split)
    imdb_dataset = load_dataset("imdb", cache_dir="./cache")

    # Then access the train/test splits
    train_data = imdb_dataset["train"]
    test_data = imdb_dataset["test"]

    train_data = train_data.shuffle(seed=0).select(range(num_samples))
    test_data = test_data.shuffle(seed=0).select(range(num_samples))

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_function(batch, T):
        # make all texts lower case
        batch['text'] = [text.lower() for text in batch['text']]
        # Tokenize and pad the text
        return tokenizer(batch['text'], padding="max_length", truncation=True, max_length=T)

    # Apply preprocessing function to tokenizing and padding/truncating text
    train_data = train_data.map(lambda batch: preprocess_function(batch, T), batched=True)
    test_data = test_data.map(lambda batch: preprocess_function(batch, T), batched=True)

    train_data = train_data.map(batched=True, remove_columns=['text', 'attention_mask', 'token_type_ids'])
    test_data = test_data.map(batched=True, remove_columns=['text', 'attention_mask', 'token_type_ids'])

    train_data = train_data.rename_column("label", "labels")
    test_data = test_data.rename_column("label", "labels")

    train_data.set_format(type='torch', columns=['input_ids', 'labels'])
    test_data.set_format(type='torch', columns=['input_ids', 'labels'])

    train_data = TensorDataset(train_data[:]['input_ids'], train_data[:]['labels'])
    test_data = TensorDataset(test_data[:]['input_ids'], test_data[:]['labels'])

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

    return train_loader, test_loader, tokenizer

def get_listops(dataset_dir, T_min, T_max, num_train=1000, num_valid=10, num_test=1000, balanced=True, batch_size=32, seed=0):
    """
    Args:
        T_min (int): Minimum length of the input sequence.
        T_max (int): Maximum length of the input sequence.
        num_train (int): Number of samples in the training set.
        num_valid (int): Number of samples in the validation set (the validation set is not used).
        num_test (int): Number of samples in the test set.
        balanced (bool): Whether to use the balanced dataset.
        batch_size (int): Batch size for the DataLoader.
    
    Returns:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        vocab (dict): Vocabulary.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Helper function to preprocess source text (remove brackets, etc.)
    def preprocess_text(text):
        text = re.sub(r'\(', '', text)  # Remove '('
        text = re.sub(r'\)', '', text)  # Remove ')'
        return text

    # Tokenize text by words (using simple space split)
    def tokenize(text, vocab):
        tokens = text.split()  # Tokenize by words (split by whitespace)
        token_ids = []
        
        for token in tokens:
            if token in vocab:
                token_ids.append(vocab[token])  # Convert token to ID
            else:
                raise ValueError(f"Token '{token}' not found in the vocabulary.")  # Raise error if token not found
        return token_ids
    
    # Manually define the vocabulary
    vocab = {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
        '[MIN': 10,
        '[MAX': 11,
        '[MED': 12,
        '[SM': 13,
        ']': 14,
        '<pad>': 15
    }

    # Define file paths
    if balanced:
        data_dir = f'{dataset_dir}/listops/balanced_Tmin={T_min}_Tmax={T_max}_num_train_samples={num_train}_num_valid_samples={num_valid}_num_test_samples={num_test}'
    else:
        data_dir = f'{dataset_dir}/listops/Tmin={T_min}_Tmax={T_max}_num_train_samples={num_train}_num_valid_samples={num_valid}_num_test_samples={num_test}'        
    
    train_dir = f'{data_dir}/basic_train.tsv'
    test_dir = f'{data_dir}/basic_test.tsv'
    
    # Load datasets (adjust to only include the first few samples for validation)
    train_data = pd.read_csv(train_dir, sep='\t', nrows=num_train)
    test_data = pd.read_csv(test_dir, sep='\t', nrows=num_test)
    
    # Preprocess and tokenize the Source column
    train_data['Source'] = train_data['Source'].apply(preprocess_text)
    test_data['Source'] = test_data['Source'].apply(preprocess_text)

    # Tokenize data into input IDs and prepare TensorDataset
    def create_tensor_dataset(data, vocab):
        inputs = [tokenize(text, vocab) for text in data['Source']]
        targets = data['Target'].tolist()
        
        # Pad sequences to the same length (to the max sequence length in the batch)
        max_length = max(len(seq) for seq in inputs)
        padded_inputs = [seq + [vocab['<pad>']] * (max_length - len(seq)) for seq in inputs]  # Use <pad> for padding
        
        return TensorDataset(torch.tensor(padded_inputs, dtype=torch.int64), torch.tensor(targets, dtype=torch.float64))

    # Create train and test datasets
    train_dataset = create_tensor_dataset(train_data, vocab)
    test_dataset = create_tensor_dataset(test_data, vocab)
    
    # Create DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Return DataLoaders and vocab
    return train_loader, test_loader, vocab  