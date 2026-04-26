import modal
import os
import shlex
import subprocess
import csv
import shutil
import re
from pathlib import Path
import yaml

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

    def _extract_exp_and_config(cli_args: list[str]) -> tuple[str, str | None]:
        exp_name_local = "cms_experiment"
        config_path_local = None
        if "--config" in cli_args:
            config_idx = cli_args.index("--config")
            if config_idx + 1 < len(cli_args):
                config_path_local = cli_args[config_idx + 1]
                cfg_path = Path(config_path_local)
                exp_name_local = cfg_path.parent.name
                if exp_name_local in {"configs", ".", ""}:
                    exp_name_local = cfg_path.stem
        return exp_name_local, config_path_local

    def _load_sweep_models(exp_name_local: str) -> list[str]:
        csv_path = Path("experiments") / exp_name_local / "model_metrics_template.csv"
        if not csv_path.exists():
            return []
        models: list[str] = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                m = str(row.get("model", "")).strip()
                if m:
                    models.append(m)
        return models

    def _sweep_overrides_from_name(model_name: str) -> dict:
        """Infer model/training overrides from template row names.

        Examples:
        - ..._160MinGRU
        - ..._mambav1_64_2_20ep
        - ..._mambav2
        """
        lower = model_name.lower()
        overrides: dict = {
            "precision": "fp16" if "fp16" in lower else "fp32",
            "backbone": "mambav1",
        }

        if "mingru" in lower:
            overrides["backbone"] = "mingru"
        elif "mambav2" in lower:
            overrides["backbone"] = "mambav2"
        elif "mambav1" in lower:
            overrides["backbone"] = "mambav1"

        # d_model from patterns like "_160MinGRU" or "_128dm_"
        m = re.search(r"_(\d+)mingru$", lower)
        if m:
            overrides["d_model"] = int(m.group(1))
        m = re.search(r"_(\d+)dm_", lower)
        if m:
            overrides["d_model"] = int(m.group(1))

        # d_model/num_layers/epochs from patterns like "_64_2_20ep"
        m = re.search(r"_(\d+)_(\d+)_(\d+)ep$", lower)
        if m:
            overrides["d_model"] = int(m.group(1))
            overrides["num_layers"] = int(m.group(2))
            overrides["epochs"] = int(m.group(3))
        else:
            # patterns like "..._30epoch"
            m = re.search(r"_(\d+)epoch$", lower)
            if m:
                overrides["epochs"] = int(m.group(1))

        return overrides

    def _train_missing_sweep_ckpt(exp_root: Path, exp_name_local: str, config_path_local: str, model_name: str) -> Path | None:
        cfg = _load_config(config_path_local)
        if not cfg:
            return None

        overrides = _sweep_overrides_from_name(model_name)
        model_cfg = dict(cfg.get("model", {}) or {})
        training_cfg = dict(cfg.get("training", {}) or {})

        if "d_model" in overrides:
            model_cfg["d_model"] = int(overrides["d_model"])
        if "num_layers" in overrides:
            model_cfg["num_layers"] = int(overrides["num_layers"])
        model_cfg["backbone"] = overrides.get("backbone", model_cfg.get("backbone", "mambav1"))

        if "epochs" in overrides:
            training_cfg["epochs"] = int(overrides["epochs"])

        cfg["model"] = model_cfg
        cfg["training"] = training_cfg
        cfg["precision"] = overrides.get("precision", cfg.get("precision", "fp32"))
        cfg["name"] = exp_name_local

        tmp_cfg = exp_root / f"_sweep_{model_name}.yaml"
        with open(tmp_cfg, "w") as f:
            yaml.safe_dump(cfg, f)

        print(
            "[INFO] Training sweep model "
            f"{model_name} (backbone={model_cfg.get('backbone')}, "
            f"d_model={model_cfg.get('d_model')}, num_layers={model_cfg.get('num_layers')}, "
            f"epochs={training_cfg.get('epochs')}, precision={cfg.get('precision')})"
        )
        _run_cmd(["python", "main.py", "--config", str(tmp_cfg), "--train-only"])

        precision = str(cfg.get("precision", "fp32"))
        trained_ckpt = exp_root / f"{exp_name_local}_final_model_{precision}.pt"
        if trained_ckpt.exists():
            return trained_ckpt
        return _find_boa_final_ckpt(exp_root, exp_name_local)

    def _run_cmd(cmd: list[str]) -> None:
        run_cmd = list(cmd)
        if run_cmd and run_cmd[0] == "python" and "-u" not in run_cmd[1:3]:
            run_cmd.insert(1, "-u")
        print(f"Running in Modal: {' '.join(run_cmd)}")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        subprocess.run(run_cmd, check=True, env=env)

    def _maybe_generate_pareto(exp_name_local: str) -> None:
        csv_path = Path("experiments") / exp_name_local / "model_metrics_template.csv"
        if not csv_path.exists():
            csv_path = Path("experiments") / exp_name_local / "model_metrics.csv"
        if not csv_path.exists():
            print(f"[INFO] Pareto plot skipped: no metrics CSV found for {exp_name_local}")
            return

        out_dir = Path("experiments") / exp_name_local / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "model_pareto_frontier.png"
        cmd = [
            "python", "pareto_plot.py",
            "--input", str(csv_path),
            "--output", str(out_path),
            "--title", f"{exp_name_local} Pareto Frontier",
            "--xlabel", "Compression Throughput (MB/s)",
            "--ylabel", "Compression Ratio",
        ]
        _run_cmd(cmd)

    def _load_config(path: str | None) -> dict:
        if path is None:
            return {}
        p = Path(path)
        if not p.exists():
            return {}
        with open(p, "r") as f:
            return yaml.safe_load(f) or {}

    def _ensure_dirs(exp_name_local: str) -> tuple[Path, Path, Path]:
        exp_root = Path("experiments") / exp_name_local
        boa_ckpt_dir = exp_root / "checkpoints" / "boa"
        hydra_ckpt_dir = exp_root / "checkpoints" / "hydra"
        boa_ckpt_dir.mkdir(parents=True, exist_ok=True)
        hydra_ckpt_dir.mkdir(parents=True, exist_ok=True)
        return exp_root, boa_ckpt_dir, hydra_ckpt_dir

    def _find_boa_final_ckpt(exp_root: Path, exp_name_local: str) -> Path | None:
        candidates = sorted(exp_root.glob(f"{exp_name_local}_final_model_*.pt"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]
        fallback = sorted(exp_root.glob("*_final_model_*.pt"), key=lambda p: p.stat().st_mtime)
        return fallback[-1] if fallback else None

    script = "main_hydra.py" if pipeline == "hydra" else "main.py"
    exp_name, config_path = _extract_exp_and_config(args)

    # BOA_SWEEP_MODELS=1 runs main.py once per model listed in
    # experiments/<exp_name>/model_metrics_template.csv using --model-path.
    sweep_models = os.environ.get("BOA_SWEEP_MODELS", "1").strip().lower() in {"1", "true", "yes"}

    if pipeline == "compare":
        if config_path is None:
            raise ValueError("compare pipeline requires BOA args to include --config <path>")

        cfg = _load_config(config_path)
        exp_root, boa_ckpt_dir, hydra_ckpt_dir = _ensure_dirs(exp_name)
        metrics_csv = exp_root / "model_metrics_template.csv"

        # 1) Train BOA first so checkpoint exists.
        boa_main_ckpt = boa_ckpt_dir / f"{exp_name}_boa_main.pt"
        if not boa_main_ckpt.exists():
            print(f"[INFO] BOA main checkpoint not found at {boa_main_ckpt}. Checking for raw checkpoint...")
            boa_source = _find_boa_final_ckpt(exp_root, exp_name)
            if boa_source is None:
                print(f"[INFO] No BOA checkpoint found. Training...")
                _run_cmd(["python", "main.py"] + list(args) + ["--train-only"])
                boa_source = _find_boa_final_ckpt(exp_root, exp_name)
            
            if boa_source is None:
                # Last resort: check if training updated the config with a path
                # we can find.
                updated_cfg = _load_config(config_path)
                model_path_cfg = updated_cfg.get("model_path")
                if model_path_cfg:
                    mp = Path(model_path_cfg)
                    if not mp.is_absolute():
                        mp = (Path(config_path).parent / mp).resolve()
                    if mp.exists():
                        boa_source = mp

            if boa_source is None:
                raise FileNotFoundError(f"Could not find BOA final checkpoint under {exp_root} even after training.")
            
            shutil.copy2(boa_source, boa_main_ckpt)
            print(f"[INFO] BOA checkpoint canonicalized: {boa_main_ckpt}")
        else:
            print(f"[INFO] BOA checkpoint already exists at {boa_main_ckpt}. Skipping training.")

        # Shared compare payload size for fair BOA vs Hydra comparison.
        # 0 means full-file comparison (default).
        shared_test_bytes = int(os.environ.get("HYDRA_TEST_BYTES", "0"))
        print(f"[INFO] Shared compare payload: {shared_test_bytes} bytes (0 = full file)")

        boa_compare_args = list(args)
        while "--model-path" in boa_compare_args:
            i = boa_compare_args.index("--model-path")
            del boa_compare_args[i:i+2]

        # If bounded compare bytes are requested, create a temporary compare config
        # with compression.file_to_compress pointing to the trimmed payload.
        if shared_test_bytes > 0 and config_path is not None:
            cfg_path = Path(config_path)
            source_path = cfg.get("compression", {}).get("file_to_compress") or cfg.get("file_path")
            if source_path is not None:
                src = Path(source_path)
                if not src.is_absolute():
                    src = (cfg_path.parent / src).resolve()
                if src.exists():
                    compare_dir = exp_root / "compare_inputs"
                    compare_dir.mkdir(parents=True, exist_ok=True)
                    compare_payload = compare_dir / f"compare_bytes_{shared_test_bytes}.bin"
                    with open(src, "rb") as rf:
                        payload = rf.read(max(0, shared_test_bytes))
                    with open(compare_payload, "wb") as wf:
                        wf.write(payload)

                    cfg_compare = dict(cfg)
                    comp = dict(cfg.get("compression", {}) or {})
                    # Use absolute path so main.py config-relative resolution cannot double-prefix.
                    comp["file_to_compress"] = str(compare_payload.resolve())
                    cfg_compare["compression"] = comp
                    compare_cfg = exp_root / "compare_config.yaml"
                    with open(compare_cfg, "w") as f:
                        yaml.safe_dump(cfg_compare, f)

                    while "--config" in boa_compare_args:
                        i = boa_compare_args.index("--config")
                        del boa_compare_args[i:i+2]
                    boa_compare_args += ["--config", str(compare_cfg)]
                    print(f"[INFO] BOA compare payload prepared: {compare_payload} ({len(payload)} bytes)")

        # 3) Run BOA compression/decompression compare using canonical checkpoint.
        _run_cmd(["python", "main.py"] + boa_compare_args + ["--model-path", str(boa_main_ckpt)])

        # 4) Train + compare Hydra on same dataset, write metrics to same CSV.
        data_path = cfg.get("file_path")
        if data_path:
            dp = Path(data_path)
            if not dp.is_absolute():
                dp = (Path(config_path).parent / dp).resolve()
            data_path = str(dp)

        hydra_epochs = os.environ.get("HYDRA_EPOCHS", "10")
        hydra_k = os.environ.get("HYDRA_K", "4")
        hydra_lr = os.environ.get("HYDRA_LR", "1e-3")
        # Default to a bounded compare payload so runs don't appear stalled.
        # Set HYDRA_TEST_BYTES=0 for full-file compress/decompress benchmarking.
        hydra_test_bytes = str(shared_test_bytes)
        hydra_use_gpu_codec = os.environ.get("HYDRA_GPU_CODEC", "1").strip().lower() in {"1", "true", "yes"}
        hydra_ckpt = hydra_ckpt_dir / f"{exp_name}_hydra_main.pt"

        hydra_args = [
            "python", "main_hydra.py",
            "--device", "cuda",
            "--epochs", str(hydra_epochs),
            "--K", str(hydra_k),
            "--lr", str(hydra_lr),
            "--test-bytes", str(hydra_test_bytes),
            "--metrics-csv", str(metrics_csv),
            "--metrics-model-name", f"{exp_name}_hydra_main",
        ]
        if hydra_ckpt.exists():
            print(f"[INFO] Hydra checkpoint already exists at {hydra_ckpt}. Skipping training.")
            hydra_args += ["--load-checkpoint", str(hydra_ckpt)]
        else:
            print(f"[INFO] Hydra checkpoint not found. Training...")
            hydra_args += ["--save-checkpoint", str(hydra_ckpt)]

        if hydra_use_gpu_codec:
            hydra_args.append("--gpu-codec")
        if data_path:
            hydra_args += ["--data-path", data_path]
        
        _run_cmd(hydra_args)

        _maybe_generate_pareto(exp_name)

    elif pipeline == "hydra":
        # For standalone hydra pipeline, check if checkpoint exists to avoid retraining if not requested.
        # We look for --save-checkpoint or --load-checkpoint in args first.
        has_ckpt_arg = "--save-checkpoint" in args or "--load-checkpoint" in args
        if not has_ckpt_arg:
            # Try to find a default hydra checkpoint
            exp_root, _, hydra_ckpt_dir = _ensure_dirs(exp_name)
            hydra_ckpt = hydra_ckpt_dir / f"{exp_name}_hydra_main.pt"
            if hydra_ckpt.exists():
                print(f"[INFO] Hydra checkpoint found at {hydra_ckpt}. Loading...")
                args += ["--load-checkpoint", str(hydra_ckpt)]
            else:
                print(f"[INFO] Hydra checkpoint not found. Will train and save to {hydra_ckpt}")
                args += ["--save-checkpoint", str(hydra_ckpt)]
        
        _run_cmd(["python", script] + args)

    elif pipeline == "boa" and sweep_models and config_path is not None:
        model_names = _load_sweep_models(exp_name)
        if model_names:
            print(f"Running BOA sweep for {len(model_names)} models from template CSV...")
            exp_root, _, _ = _ensure_dirs(exp_name)
            for model_name in model_names:
                ckpt = Path("experiments") / exp_name / f"{model_name}.pt"
                if not ckpt.exists():
                    print(f"[INFO] Checkpoint not found for model '{model_name}': {ckpt}. Training it now.")
                    trained = _train_missing_sweep_ckpt(exp_root, exp_name, config_path, model_name)
                    if trained is None or not trained.exists():
                        print(f"[WARN] Could not train checkpoint for model '{model_name}' (skipping)")
                        continue
                    shutil.copy2(trained, ckpt)
                    print(f"[INFO] Sweep checkpoint ready: {ckpt}")

                base_args = list(args)
                while "--model-path" in base_args:
                    i = base_args.index("--model-path")
                    del base_args[i:i+2]

                cmd = ["python", script] + base_args + ["--model-path", str(ckpt)]
                _run_cmd(cmd)
            _maybe_generate_pareto(exp_name)
        else:
            _run_cmd(["python", script] + args)
    else:
        _run_cmd(["python", script] + args)

    # Collect all experiment result files to send back to the local machine
    exp_dir = f"experiments/{exp_name}"
    synced_files = {}
    
    if os.path.exists(exp_dir):
        for root, _, files in os.walk(exp_dir):
            for file in files:
                valid_extensions = (".pt", ".yaml", ".boa", ".bin", ".png", ".pdf", ".lzma", ".zlib", ".csv", ".json")
                if file.endswith(valid_extensions):
                    filepath = os.path.join(root, file)
                    with open(filepath, "rb") as f:
                        synced_files[filepath] = f.read()
                        
    return synced_files

@app.local_entrypoint()
def main():
    # Choose pipeline: "compare" (default), "boa", or "hydra"
    pipeline = os.environ.get("BOA_PIPELINE", "compare").strip().lower()
    if pipeline not in {"boa", "hydra", "compare"}:
        raise ValueError("BOA_PIPELINE must be 'compare', 'boa', or 'hydra'")

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
