import torch
import argparse
import os
import io

def get_size_of_state_dict(state_dict):
    """
    Calculate the size of a state dictionary in megabytes (MB).
    
    Args:
        state_dict (dict): The state dictionary to measure.
        
    Returns:
        float: Size in MB.
    """
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    size_mb = len(buffer.getvalue()) / (1024 ** 2)  # Convert bytes to MB
    return size_mb

def main():
    parser = argparse.ArgumentParser(description="Inspect checkpoint components and their sizes.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file.")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path

    if not os.path.isfile(checkpoint_path):
        print(f"Checkpoint file does not exist: {checkpoint_path}")
        return

    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load the checkpoint on CPU to save GPU memory
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Display all keys in the checkpoint
    print(f"Checkpoint contains the following keys: {list(checkpoint.keys())}\n")
    
    # Iterate through each key and calculate its size
    for key in checkpoint:
        state = checkpoint[key]
        
        if isinstance(state, dict):
            size_mb = get_size_of_state_dict(state)
            print(f"Size of '{key}': {size_mb:.2f} MB")
        else:
            # For non-dict entries like 'args', handle accordingly
            try:
                buffer = io.BytesIO()
                torch.save(state, buffer)
                size_mb = len(buffer.getvalue()) / (1024 ** 2)
                print(f"Size of '{key}': {size_mb:.2f} MB")
            except Exception as e:
                print(f"Could not determine size of '{key}': {e}")

if __name__ == "__main__":
    main()
