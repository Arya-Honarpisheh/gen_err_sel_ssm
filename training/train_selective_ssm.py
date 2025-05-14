import time
import csv
import numpy as np
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy
from einops import einsum
from models.selective_ssm import SentimentModel
from sklearn.metrics import classification_report


class CollapseAvoidLoss(torch.nn.Module):
    # from Yannik Keller at https://yannikkeller.substack.com/p/solving-vanishing-gradients-from?r=3avwpj&triedRedirect=true
    def __init__(self, min_std=0.1, factor=10):
        """
        Prevent output standard deviation from collapsing below a threshold.
        Args:
            min_std (float): Minimum allowed standard deviation.
            factor (float): Scaling factor for the penalty.
        """
        
        super(CollapseAvoidLoss, self).__init__()
        self.min_std = min_std
        self.factor = factor

    def forward(self, logits):
        """
        Compute the collapse avoidance loss.
        Args:
            logits (torch.Tensor): Model output logits before applying sigmoid.
        Returns:
            torch.Tensor: Penalty loss to prevent collapse.
        """
        
        std = torch.std(torch.sigmoid(logits))
        return torch.clamp((self.min_std - std) * self.factor, min=0.0)
    
    
# Initialize the CSV file with headers
def initialize_csv(filename):
    headers = ['epoch', 's_A', 'L_p', 'B_q', 'B_B', 'M_B', 'B_C', 'M_C', 'B_A', 'Loss']
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

# Append data for each epoch
def log_to_csv(filename, epoch, s_A, L_p, B_q, B_B, M_B, B_C, M_C, B_A, loss):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, s_A, L_p, B_q, B_B, M_B, B_C, M_C, B_A, loss])
        
def calculate_params(model):
    """
    Calculate the parameters of the SSM block.  
    Args:
        model: Model to calculate the parameters.
    Returns:
        s_A (float): Stability margin of matrix A.
        L_p (float): L2 norm of the p_delta parameter.
        B_q (float): L2 norm of the q_delta parameter.
        B_B (float): L2 norm of the W_B parameter.
        M_B (float): L1 norm of the W_B parameter.
        B_C (float): L2 norm of the W_C parameter.
        M_C (float): L1 norm of the W_C parameter.
        B_A (float): L2 norm of the A matrix (rows of A are the diagonal for each A_j).
    """
    
    if model.__class__.__name__ == 'SentimentModel':
        ssm = model.SSM.ssm
    elif model.__class__.__name__ == 'MultiClassModel':
        ssm = model.ssm
    else:
        raise ValueError(f"Invalid model class: {model.__class__.__name__}")
    
    s_A = -ssm.A.detach().max().item()
    L_p = torch.norm(ssm.p_delta, p=2).item()
    B_q = torch.norm(ssm.q_delta, p=2).item()
    B_B = torch.norm(ssm.W_B, p=2).item()
    M_B = torch.sum(torch.abs(ssm.W_B)).item()
    B_C = torch.norm(ssm.W_C, p=2).item()
    M_C = torch.sum(torch.abs(ssm.W_C)).item()
    B_A = torch.max(torch.abs(ssm.A)).item()
    
    return s_A, L_p, B_q, B_B, M_B, B_C, M_C, B_A


# Training Loop
def train_ssm_block(device, task, T, s_A, d, N, num_classes, use_delta=True, fix_sA=True, data_loader=None, criterion=None, num_epochs=10, learning_rate=1e-3, weight_decay=1e-4, tokenizer=None, vocab=None, log_file=None):
    """
    Train the SSM block for the given task.
    Args:
        device (torch.device): Device to be used.
        log_file (str): Path to the CSV file to log the parameters.
        task (str): Task to be trained on.
        T (int): Length of the input sequence.
        s_A (float): Stability margin of matrix A.
        d (int): Number of input channels.
        N (int): Number of states per channel.
        num_classes (int): Number of classes for the classification task.
        use_delta (bool): Whether to use the discretization parameter.
        fix_sA (bool): Whether to fix the stability margin s_A for the first entry of A.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        criterion (torch.nn.Module): Loss criterion.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        tokenizer (transformers.AutoTokenizer): Tokenizer for the dataset (For IMDb).
        vocab (dict): Vocabulary for the dataset (For ListOps).
    Returns:
        losses (list): List of losses for each epoch.
        models (models.selective_ssm.SentimentModel): Trained model.
    """
    
    if data_loader==None or criterion==None:
        raise ValueError(f"Specify data_loader and criterion.")

    if task == 'imdb':
        vocab_size = tokenizer.vocab_size
        model = SentimentModel(vocab_size, d, N, s_A, use_delta, fix_sA, device).to(device)
        accuracy_metric = BinaryAccuracy().to(device)  # Binary classification task
    elif task == 'majority':
        model = SentimentModel(2, d, N, s_A, use_delta, fix_sA, device).to(device)
        accuracy_metric = BinaryAccuracy().to(device)  # Binary classification task
    else:
        raise ValueError(f"Invalid task: {task}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor = 0.9, patience=5,
                                                           threshold=0.01, verbose=False)

# Training loop
    model.train()
    losses = []
    
    # Initialize the CSV file for the log
    if log_file is not None:
        initialize_csv(log_file)
        s_A, L_p, B_q, B_B, M_B, B_C, M_C, B_A = calculate_params(model)
        log_to_csv(log_file, 0, s_A, L_p, B_q, B_B, M_B, B_C, M_C, B_A, 0)  
    
    accuracy_epochs = 5  # You can adjust this value for how often you want to log accuracy

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0
        accuracy_metric.reset()  # Reset the accuracy metric at the start of each epoch
        
        # print("sA: ", -model.SSM.ssm.A.detach().max())
        # print("L_p: ", torch.norm(model.SSM.ssm.p_delta, p=2))
        # print("B_q: ", torch.norm(model.SSM.ssm.q_delta, p=2))
        
        # Print the stability margin for each epoch
        # print(f"Stability Margin: {-model.SSM.ssm.A.detach().max():.2f}")
        
        for batch in data_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
                                             
            # Check the ouput of model
            if torch.isnan(outputs).any():
                raise ValueError("Model output is NaN!")
            if torch.isinf(outputs).any():
                raise ValueError("Model output is infinite!")

            ##### To not stuck in local minima, we add a penalty term to the loss function #####
            loss = criterion(outputs, y_batch.float())
            std_loss = CollapseAvoidLoss(min_std=0.1, factor=10)
            loss += std_loss(outputs)
            ###############################################################################

            # Check loss value
            if torch.isnan(loss):
                raise ValueError(f"NaN detected in loss!")

            loss.backward()

            # Check the gradients
            for name, param in model.named_parameters():
                if torch.isnan(param.grad).any():
                    raise ValueError(f"NaN detected in gradient of {name}!")
                
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            epoch_loss += loss.item()
            
            # Update accuracy metric
            accuracy_metric.update(outputs, y_batch)
        
        avg_loss = epoch_loss / len(data_loader)
        scheduler.step(avg_loss)

        current_lr = scheduler.get_last_lr()[0]
        losses.append(avg_loss)

        epoch_end_time = time.time()  # Track the end time for the epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}, Learning Rate: {current_lr:.6f}, Time: {epoch_end_time - epoch_start_time:.2f} seconds")
        
        # Compute the stability margin and norms for the last epoch
        s_A, L_p, B_q, B_B, M_B, B_C, M_C, B_A = calculate_params(model)
        if log_file is not None:
            log_to_csv(log_file, epoch+1, s_A, L_p, B_q, B_B, M_B, B_C, M_C, B_A, avg_loss)
        
        # Log and print accuracy every accuracy_epochs epochs
        if (epoch + 1) % accuracy_epochs == 0:
            accuracy = accuracy_metric.compute()
            print(f"Epoch {epoch+1}: Training Accuracy: {accuracy:.4f}")

            # Check if it's multi-class by looking at output shape
            if outputs.ndim == 2 and outputs.shape[1] > 1:
                y_pred = outputs.argmax(dim=1)
                y_pred = y_pred.detach().cpu().numpy()
                y_true = y_batch.detach().cpu().numpy()
                print(classification_report(y_true, y_pred))

    
    return losses, model
