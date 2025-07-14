import glob
import re
import os

def find_latest_checkpoint():
    """Find the checkpoint file with the highest iteration number"""
    checkpoint_pattern = "./checkpoints/elevated_damped_oscillator-*.weights.h5"
    checkpoint_files = glob.glob(checkpoint_pattern)

    if not checkpoint_files:
        print("No checkpoint files found.")
        return None

    # Extract iteration numbers from filenames
    iterations = []
    for file_path in checkpoint_files:
        filename = os.path.basename(file_path)
        match = re.search(r"-(\d+)\.weights\.h5$", filename)
        if match:
            iterations.append((int(match.group(1)), file_path))

    if not iterations:
        print("No valid checkpoint files with iteration numbers found.")
        return None

    # Sort by iteration number and get the highest one
    iterations.sort(reverse=True)
    highest_iter, latest_file = iterations[0]

    print(f"Found latest checkpoint at iteration {highest_iter}: {latest_file}")
    return latest_file
