import numpy as np
import random

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from datasets import load_dataset


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
    noise_mask = (np.random.rand(num_samples, T) < 0.1) & (inputs_train == 1)
    inputs_train[noise_mask] = 0

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

def get_imdb(T, num_samples=1000, batch_size=16):
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
    
    train_data = load_dataset('imdb', split='train')
    test_data = load_dataset('imdb', split='test')

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

    train_data = TensorDataset(train_data['input_ids'], train_data['labels'])
    test_data = TensorDataset(test_data['input_ids'], test_data['labels'])

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

    return train_loader, test_loader, tokenizer