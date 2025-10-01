import os
import numpy as np
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
from training.train_selective_ssm import train_ssm_block
from training.load_datasets import get_imdb, create_majority_dataset, get_listops
from models.selective_ssm import SentimentModel, MultiClassModel

def parse_args():
    parser = argparse.ArgumentParser(description="Stability Margin and Length Independence Experiments")
    parser.add_argument('--devID', type=int, default=0, help="GPU ID to use")
    parser.add_argument('--experiment', choices=['stability_margin_T', 'length_independence'], required=True, help="Choose the experiment: stability_margin or length_independence")
    parser.add_argument('--dataset', choices=['imdb', 'majority', 'listops'], required=True, help="Choose the dataset")
    parser.add_argument('--T_values', type=int, nargs='+', default=[100], help="List of sequence lengths to test")
    parser.add_argument('--T_var', type=int, default=5, help="+- variation for T value (Tmin/Tmax for ListOps)")
    parser.add_argument('--s_A_values', type=float, nargs='+', default=[0], help="Stability margin for SSM")
    parser.add_argument('--N', type=int, default=25, help="Number of states per channel in SSM")
    parser.add_argument('--d', type=int, default=10, help="Number of input channels")
    parser.add_argument('--num_train', type=int, default=1000, help="Number of training samples")
    parser.add_argument('--num_valid', type=int, default=0, help="Number of validation samples")
    parser.add_argument('--num_test', type=int, default=1000, help="Number of test samples")
    parser.add_argument('--balanced', type=str, default="false", help="Use balanced dataset for ListOps")
    parser.add_argument('--num_epochs', type=int, default=5, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--use_delta', type=str, default="true", help="Use delta in SSM")
    parser.add_argument('--fix_sA', type=str, default="true", help="Fix s_A for the first entry of the matrix A")
    parser.add_argument('--save_results', type=str, default="true", help="Save the trained model and results")
    parser.add_argument('--model_path', type=str, default='', help="Path to the trained model weights")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    return parser.parse_args()

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility
    torch.backends.cudnn.benchmark = False     # Disable auto-tuning for determinism

def run_experiment(args):
    # Check if GPU is available and set the device with the specified GPU ID
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(args.devID)
        print(f"Using GPU: {torch.cuda.get_device_name(args.devID)} with ID {args.devID}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Create necessary directories
    os.makedirs(f'results/{args.experiment}/{args.dataset}/epochs', exist_ok=True)
    os.makedirs(f'./models/trained/{args.experiment}/{args.dataset}', exist_ok=True)\
    # Create a filename with the experiment details
    filename = f'N_{args.N}_d_{args.d}_m_{args.num_train}_ep_{args.num_epochs}_bs_{args.batch_size}_lr_{args.learning_rate}_wd_{args.weight_decay}_delta_{args.use_delta}_seed_{args.seed}'
    # Convert the string flag arguments to boolean
    args.use_delta = args.use_delta.lower() == 'true'
    args.fix_sA = args.fix_sA.lower() == 'true'
    args.save_results = args.save_results.lower() == 'true'
    args.balanced = args.balanced.lower() == 'true'

    # Print the initial details
    print("--------------------------------------------------")
    print(f"Running {args.experiment} experiment on {args.dataset} dataset...")
    print(f"Sequence Lengths: {args.T_values}")
    print(f"Stability Margins: {args.s_A_values}")
    print(f"Number of States per Channel (N): {args.N}")
    print(f"Number of Input Channels (d): {args.d}")
    print(f"Learing Rate: {args.learning_rate}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Epochs: {args.num_epochs}")
    print(f"Use discretization with delta: {args.use_delta}")
    print(f"Save Results: {args.save_results}")
    print("--------------------------------------------------")

    results = {}

    # Training and testing
    for T in args.T_values:

        # Set seed for reproducibility
        set_seed(args.seed)
        
        # Select dataset and load data
        if args.dataset == 'imdb':
            print("Using IMDB Dataset...")
            criterion = nn.BCEWithLogitsLoss()
            num_classes = 2
            train_loader, _, tokenizer = get_imdb(T=T, num_samples=args.num_train, batch_size=args.batch_size, seed=args.seed)
        elif args.dataset == 'majority':
            print("Using Sparse Majority Dataset...")
            criterion = nn.BCEWithLogitsLoss()
            num_classes = 2
            train_loader, _ = create_majority_dataset(T=T, num_samples=args.num_train, batch_size=args.batch_size, seed=args.seed)
        elif args.dataset == 'listops':
            print("Using ListOps Dataset...")
            criterion = nn.CrossEntropyLoss()
            num_classes = 10
            train_loader, _, vocab = get_listops(dataset_dir='./dataset', T_min=T-args.T_var, T_max=T+args.T_var, num_train=args.num_train, num_valid=args.num_valid, num_test=args.num_test, balanced=args.balanced, batch_size=args.batch_size, seed=args.seed)
        
        
        for s_A in args.s_A_values:
            print(f"Processing T={T}, s_A={s_A}...")
            
            # Create a log file to save the results
            if args.save_results:
                log_file = f'results/{args.experiment}/{args.dataset}/epochs/log_T_{T}_sA_{s_A}_{filename}.csv'
            else:
                log_file = None
            
            # Load model weights if specified, train new model otherwise
            if args.model_path == '':
                # Create the model path
                model_path = f'./models/trained/{args.experiment}/{args.dataset}/T_{T}_sA_{s_A}_{filename}.pth'
                # Train the model
                print(f"Training new model with T={T}, s_A={s_A}")
            else:
                model_path = args.model_path
                print(f"Loading model from {model_path}, training with T={T}, s_A={s_A}")
                if args.dataset == 'imdb':
                    vocab_size = tokenizer.vocab_size
                    model = SentimentModel(vocab_size, args.d, args.N, s_A, args.use_delta, device).to(device)
                elif args.dataset == 'majority':
                    model = SentimentModel(2, args.d, args.N, s_A, args.use_delta, device).to(device)
                elif args.dataset == 'listops':
                    model = MultiClassModel(len(vocab), args.d, args.N, s_A, num_classes, args.use_delta, device).to(device)
                                    
                model.load_state_dict(torch.load(model_path, weights_only=True))
            
            losses, model = train_ssm_block(device=device, 
                                            task=args.dataset, 
                                            T=T, s_A=s_A, d=args.d, N=args.N, 
                                            num_classes=num_classes,
                                            use_delta=args.use_delta, 
                                            fix_sA=args.fix_sA,
                                            data_loader=train_loader,
                                            criterion=criterion,
                                            num_epochs=args.num_epochs,
                                            learning_rate=args.learning_rate, 
                                            weight_decay=args.weight_decay, 
                                            tokenizer=tokenizer if ( args.dataset == 'imdb') else None, 
                                            vocab=vocab if args.dataset =='listops' else None, 
                                            log_file=log_file)
            
            # Save the training loss wrt the experiment we are conducting
            if args.experiment == 'length_independence':
                results[T] = losses[-1]
            elif args.experiment == 'stability_margin':
                results[s_A] = losses[-1]
            else:
                ValueError(f'Invalid experiment: {args.experiment}.')
                
            # Save the trained model
            if args.save_results and args.experiment == 'length_independence':
                torch.save(model.state_dict(), model_path)

            # Save training loss plot
            # plt.figure()
            # plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-')
            # plt.xlabel('Epochs')
            # plt.ylabel('Loss')
            # plt.title(f'Training Loss - T={T}, s_A={s_A}, N={args.N}')
            # if args.save_results: plt.savefig(f'results/{args.experiment}/{args.dataset}/epochs/T{T}_sA_{s_A}_{filename}.png')
            # plt.close()

    print("Training complete!")

if __name__ == "__main__":
    # Parse the command line arguments
    args = parse_args()
    # Run the experiment with the parsed arguments
    run_experiment(args)
