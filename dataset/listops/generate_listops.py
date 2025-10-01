import listops
import os
import subprocess

def generate_listops_dataset(main_dir, T_min, T_max, num_train, num_valid, num_test):
    """
    Generates a ListOps dataset with specified parameters.
    
    Args:
        main_dir (str): Base directory where datasets will be stored.
        T_min (int): Minimum sequence length.
        T_max (int): Maximum sequence length.
        num_train (int): Number of training samples.
        num_test (int): Number of testing samples.
    """
    # Create base dataset directory if it does not exist
    main_dir = os.path.expanduser(main_dir)
    dataset_root = os.path.abspath(main_dir)
    os.makedirs(dataset_root, exist_ok=True)
    
    # Define dataset-specific folder name
    dataset_name = f"Tmin={T_min}_Tmax={T_max}_num_train_samples={num_train}_num_valid_samples={num_valid}_num_test_samples={num_test}"
    dataset_path = os.path.join(dataset_root, dataset_name)
    os.makedirs(dataset_path, exist_ok=True)
    
    # Run the dataset generation script
    cmd = (
        f"python ./dataset/listops/listops.py "
        f"--output_dir={dataset_path} "
        f"--min_length={T_min} --max_length={T_max} "
        f"--num_train_samples={num_train} --num_valid_samples={num_valid} --num_test_samples={num_test}"
    )
    subprocess.run(cmd, shell=True, check=True)
    
    print(f"Dataset stored in: {dataset_path}")


def main():
    main_dir = './dataset/listops'
    T_values = range(100, 1001, 100)
    T_dev = 5  # Deviation from T_value (+- for min/max seq)
    num_train_samples = 5000
    num_valid_samples = 0
    num_test_samples = 5000
    
    for T in T_values:
        generate_listops_dataset(
            main_dir=main_dir,
            T_min=T-T_dev,
            T_max=T+T_dev,
            num_train=num_train_samples,
            num_valid=num_valid_samples,
            num_test=num_test_samples
        )
        print(f"\nDataset for T={T} with {num_train_samples} train, {num_valid_samples} validation, and {num_test_samples} test samples is generated.")
    
    print("\nAll dataset generations are complete!")

if __name__ == "__main__":
    main()