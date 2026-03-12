import modal
import os
import subprocess

app = modal.App("boa-constrictor")

# Base image with Python 3.11, Nvidia CUDA Compiler (NVCC) and PyTorch/dependencies.
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(
        ".", 
        remote_path="/app",
        ignore=[".git*", ".venv", "__pycache__"]
    )
)

@app.function(image=image, gpu="T4", timeout=86400)
def train_and_sync(args: list[str]):
    os.chdir("/app")
    
    cmd = ["python", "main.py"] + args
    print(f"Running in Modal: {' '.join(cmd)}")
    
    # Execute the training pipeline in the container
    subprocess.run(cmd, check=True)
    
    # Figure out the experiment name from the config path to sync the outputs
    exp_name = "cms_experiment"
    if "--config" in args:
        config_idx = args.index("--config")
        if config_idx + 1 < len(args):
            exp_name = os.path.basename(os.path.dirname(args[config_idx + 1]))
            # If standard config path like config/my_exp.yaml
            if exp_name == "configs" or exp_name == ".":
                exp_name = os.path.splitext(os.path.basename(args[config_idx + 1]))[0]

    # Collect all experiment result files to send back to the local machine
    exp_dir = f"experiments/{exp_name}"
    synced_files = {}
    
    if os.path.exists(exp_dir):
        for root, _, files in os.walk(exp_dir):
            for file in files:
                valid_extensions = (".pt", ".yaml", ".boa", ".bin", ".png", ".pdf", ".lzma", ".zlib")
                if file.endswith(valid_extensions):
                    filepath = os.path.join(root, file)
                    with open(filepath, "rb") as f:
                        synced_files[filepath] = f.read()
                        
    return synced_files

@app.local_entrypoint()
def main():
    # Pass arguments needed to run main.py inside the container
    args = ["--config", "experiments/cms_experiment/cms_experiment.yaml"]
    
    print("Dispatching training job to Modal...")
    results = train_and_sync.remote(args)
    
    print(f"\nJob completed! Syncing {len(results)} output files back to local workspace...")
    for filepath, data in results.items():
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(data)
        print(f"Saved {filepath}")
        
    print("Done!")
