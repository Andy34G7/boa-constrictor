import modal
import os
import shlex
import subprocess
from pathlib import Path

app = modal.App("boa-constrictor")

# Base image with Python 3.11, Nvidia CUDA Compiler (NVCC) and PyTorch/dependencies.
# We install torch first, then mamba-ssm/causal-conv1d from pre-built GitHub release wheels
# to avoid source compilation failures in the Modal build environment.
TORCH_VERSION = "2.4.0+cu121"
CAUSAL_CONV1D_WHEEL = (
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/"
    "causal_conv1d-1.5.2%2Bcu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)
MAMBA_SSM_WHEEL = (
    "https://github.com/state-spaces/mamba/releases/download/v2.2.5/"
    "mamba_ssm-2.2.5%2Bcu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "ninja-build")
    # 1. Install PyTorch first so mamba can find it during its own install
    .pip_install(
        f"torch=={TORCH_VERSION}",
        "torchvision==0.19.0+cu121",
        extra_index_url="https://download.pytorch.org/whl/cu121"
    )
    # 2. Install mamba-ssm and causal-conv1d from pre-built wheels (no source compile needed)
    .pip_install(CAUSAL_CONV1D_WHEEL, MAMBA_SSM_WHEEL)
    # 3. Install remaining requirements (excluding torch, mamba-ssm, causal-conv1d)
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(
        ".", 
        remote_path="/app",
        ignore=[".git*", ".venv", "__pycache__"]
    )
)

@app.function(image=image, gpu="T4", timeout=86400)
def train_and_sync(args: list[str], pipeline: str = "boa"):
    os.chdir("/app")

    script = "main_hydra.py" if pipeline == "hydra" else "main.py"
    cmd = ["python", script] + args
    print(f"Running in Modal: {' '.join(cmd)}")
    
    # Execute the training pipeline in the container
    subprocess.run(cmd, check=True)
    
    # Figure out the experiment name from the config path to sync the outputs
    exp_name = "cms_experiment"
    if "--config" in args:
        config_idx = args.index("--config")
        if config_idx + 1 < len(args):
            cfg_path = Path(args[config_idx + 1])
            exp_name = cfg_path.parent.name
            # If config path is flat like configs/my_exp.yaml
            if exp_name in {"configs", ".", ""}:
                exp_name = cfg_path.stem

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
    # Choose pipeline: "boa" (default) or "hydra"
    pipeline = os.environ.get("BOA_PIPELINE", "boa").strip().lower()
    if pipeline not in {"boa", "hydra"}:
        raise ValueError("BOA_PIPELINE must be 'boa' or 'hydra'")

    # Args passed to selected script inside the container.
    # - BOA pipeline uses config YAML.
    # - Hydra pipeline uses main_hydra.py flags.
    if pipeline == "hydra":
        args = [
            "--device", "cuda",
            "--epochs", "10",
            "--K", "4",
            "--save-checkpoint", "experiments/cms_experiment/hydra_final_model.pt",
        ]
    else:
        args = ["--config", "experiments/cms_experiment/cms_experiment.yaml"]

    # Optional override for script arguments without editing this file.
    # Example:
    #   BOA_ARGS='--device cuda --epochs 20 --K 8 --save-checkpoint experiments/cms_experiment/hydra_k8.pt'
    args_override = os.environ.get("BOA_ARGS", "").strip()
    if args_override:
        args = shlex.split(args_override)
    
    print(f"Dispatching {pipeline} job to Modal...")
    results = train_and_sync.remote(args, pipeline=pipeline)
    
    print(f"\nJob completed! Syncing {len(results)} output files back to local workspace...")
    for filepath, data in results.items():
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(data)
        print(f"Saved {filepath}")
        
    print("Done!")
